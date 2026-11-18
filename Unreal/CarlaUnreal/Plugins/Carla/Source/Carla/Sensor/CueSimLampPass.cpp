// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla/Sensor/CueSimLampPass.h"
#include "Carla.h"
#include "Carla/Sensor/CueSimLampPassSensor.h"

#include <util/ue-header-guard-begin.h>
#include "Async/Async.h"
#include "DataDrivenShaderPlatformInfo.h"
#include "Dom/JsonObject.h"
#include "GlobalShader.h"
#include "HAL/IConsoleManager.h"
#include "MaterialDomain.h"
#include "MeshMaterialShader.h"
#include "MeshPassProcessor.inl"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "PrimitiveSceneInfo.h"
#include "PrimitiveSceneProxy.h"
#include "RenderCore.h"
#include "RenderGraphUtils.h"
#include "RHIGPUReadback.h"
#include "SceneInterface.h"
#include "SceneManagement.h"
#include "SceneRendererInterface.h"
#include "SceneTexturesConfig.h"
#include "SceneView.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "ShaderParameterStruct.h"
#include "SimpleMeshDrawCommandPass.h"
#include "TextureResource.h"
#include <util/ue-header-guard-end.h>

#include <algorithm>
#include <chrono>
#include <thread>

// =============================================================================
// -- Shaders ------------------------------------------------------------------
// =============================================================================

namespace
{
  /// Lamp materials are emissive surface materials drawn by static meshes.
  bool ShouldCompileLampShader(const FMeshMaterialShaderPermutationParameters &Parameters)
  {
    static const FName LocalVertexFactory(TEXT("FLocalVertexFactory"));
    return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5) &&
        Parameters.MaterialParameters.MaterialDomain == MD_Surface &&
        Parameters.MaterialParameters.bHasEmissiveColorConnected &&
        Parameters.VertexFactoryType->GetFName() == LocalVertexFactory;
  }

  struct FCueSimLampElementData : public FMeshMaterialShaderElementData
  {
    uint32 ActorId = 0;
    FVector4f ToLocal0 = FVector4f(1, 0, 0, 0);
    FVector4f ToLocal1 = FVector4f(0, 1, 0, 0);
    FVector4f ToLocal2 = FVector4f(0, 0, 1, 0);
    FRHITexture *Mask = nullptr;
  };
}

class FCueSimLampVS : public FMeshMaterialShader
{
  DECLARE_SHADER_TYPE(FCueSimLampVS, MeshMaterial);

public:
  FCueSimLampVS() = default;
  FCueSimLampVS(const ShaderMetaType::CompiledShaderInitializerType &Initializer)
    : FMeshMaterialShader(Initializer)
  {
    PassUniformBuffer.Bind(Initializer.ParameterMap,
        FSceneTextureUniformParameters::FTypeInfo::GetStructMetadata()->GetShaderVariableName());
  }

  static bool ShouldCompilePermutation(const FMeshMaterialShaderPermutationParameters &Parameters)
  {
    return ShouldCompileLampShader(Parameters);
  }
};

class FCueSimLampPS : public FMeshMaterialShader
{
  DECLARE_SHADER_TYPE(FCueSimLampPS, MeshMaterial);

public:
  FCueSimLampPS() = default;
  FCueSimLampPS(const ShaderMetaType::CompiledShaderInitializerType &Initializer)
    : FMeshMaterialShader(Initializer)
  {
    PassUniformBuffer.Bind(Initializer.ParameterMap,
        FSceneTextureUniformParameters::FTypeInfo::GetStructMetadata()->GetShaderVariableName());
    ActorIdParameter.Bind(Initializer.ParameterMap, TEXT("CueSimActorId"));
    ToLocal0Parameter.Bind(Initializer.ParameterMap, TEXT("CueSimToLocal0"));
    ToLocal1Parameter.Bind(Initializer.ParameterMap, TEXT("CueSimToLocal1"));
    ToLocal2Parameter.Bind(Initializer.ParameterMap, TEXT("CueSimToLocal2"));
    MaskTextureParameter.Bind(Initializer.ParameterMap, TEXT("CueSimMaskTexture"));
    MaskSamplerParameter.Bind(Initializer.ParameterMap, TEXT("CueSimMaskSampler"));
  }

  static bool ShouldCompilePermutation(const FMeshMaterialShaderPermutationParameters &Parameters)
  {
    return ShouldCompileLampShader(Parameters);
  }

  void GetShaderBindings(
      const FScene *Scene,
      ERHIFeatureLevel::Type FeatureLevel,
      const FPrimitiveSceneProxy *PrimitiveSceneProxy,
      const FMaterialRenderProxy &MaterialRenderProxy,
      const FMaterial &Material,
      const FCueSimLampElementData &ShaderElementData,
      FMeshDrawSingleShaderBindings &ShaderBindings) const
  {
    FMeshMaterialShader::GetShaderBindings(Scene, FeatureLevel, PrimitiveSceneProxy,
        MaterialRenderProxy, Material, ShaderElementData, ShaderBindings);
    ShaderBindings.Add(ActorIdParameter, ShaderElementData.ActorId);
    ShaderBindings.Add(ToLocal0Parameter, ShaderElementData.ToLocal0);
    ShaderBindings.Add(ToLocal1Parameter, ShaderElementData.ToLocal1);
    ShaderBindings.Add(ToLocal2Parameter, ShaderElementData.ToLocal2);
    // Point sampling: the kernel only accepts exact mask cell interiors.
    ShaderBindings.AddTexture(MaskTextureParameter, MaskSamplerParameter,
        TStaticSamplerState<SF_Point, AM_Wrap, AM_Wrap, AM_Wrap>::GetRHI(),
        ShaderElementData.Mask);
  }

  LAYOUT_FIELD(FShaderParameter, ActorIdParameter);
  LAYOUT_FIELD(FShaderParameter, ToLocal0Parameter);
  LAYOUT_FIELD(FShaderParameter, ToLocal1Parameter);
  LAYOUT_FIELD(FShaderParameter, ToLocal2Parameter);
  LAYOUT_FIELD(FShaderResourceParameter, MaskTextureParameter);
  LAYOUT_FIELD(FShaderResourceParameter, MaskSamplerParameter);
};

IMPLEMENT_MATERIAL_SHADER_TYPE(, FCueSimLampVS, TEXT("/Plugin/Carla/CueSimLampPass.usf"), TEXT("VSMain"), SF_Vertex);
IMPLEMENT_MATERIAL_SHADER_TYPE(, FCueSimLampPS, TEXT("/Plugin/Carla/CueSimLampPass.usf"), TEXT("PSMain"), SF_Pixel);

namespace
{
  /// Must match FCueSimLampBox in CueSimLampClassify.usf.
  struct FCueSimLampBoxGPU
  {
    FVector4f Row0;
    FVector4f Row1;
    FVector4f Row2;
    FVector4f ExtentId;
  };
}

class FCueSimLampClassifyCS : public FGlobalShader
{
  DECLARE_GLOBAL_SHADER(FCueSimLampClassifyCS);
  SHADER_USE_PARAMETER_STRUCT(FCueSimLampClassifyCS, FGlobalShader);

  BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
    SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
    SHADER_PARAMETER_RDG_TEXTURE(Texture2D<uint4>, CueSimLampIds)
    SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, CueSimLampEmissive)
    SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, CueSimLampLocal)
    SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, CueSimSceneDepth)
    SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FCueSimLampBox>, CueSimBoxes)
    SHADER_PARAMETER(uint32, CueSimNumBoxes)
    SHADER_PARAMETER(FIntPoint, CueSimViewMin)
    SHADER_PARAMETER(FIntPoint, CueSimViewSize)
    SHADER_PARAMETER(float, CueSimDepthToleranceCm)
    SHADER_PARAMETER(float, CueSimDepthToleranceRelative)
    SHADER_PARAMETER(float, CueSimEmittingThreshold)
    SHADER_PARAMETER(uint32, CueSimMaxRecords)
    SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, CueSimRecords)
    SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint>, CueSimCounter)
  END_SHADER_PARAMETER_STRUCT()

  static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters &Parameters)
  {
    return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
  }
};

IMPLEMENT_GLOBAL_SHADER(FCueSimLampClassifyCS, "/Plugin/Carla/CueSimLampClassify.usf", "MainCS", SF_Compute);

// =============================================================================
// -- Mesh pass ----------------------------------------------------------------
// =============================================================================

namespace
{
  BEGIN_SHADER_PARAMETER_STRUCT(FCueSimLampMeshPassParameters, )
    SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
    SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneUniformParameters, Scene)
    SHADER_PARAMETER_STRUCT_INCLUDE(FInstanceCullingDrawParams, InstanceCullingDrawParams)
    RENDER_TARGET_BINDING_SLOTS()
  END_SHADER_PARAMETER_STRUCT()

  class FCueSimLampMeshProcessor : public FMeshPassProcessor
  {
  public:
    FCueSimLampMeshProcessor(const FScene *Scene, const FSceneView *View,
        FMeshPassDrawListContext *Context)
      : FMeshPassProcessor(EMeshPass::Num, Scene, View->GetFeatureLevel(), View, Context)
    {
      RenderState.SetBlendState(TStaticBlendState<>::GetRHI());
      // Nearest lamp fragment only; the scene depth is compared afterwards.
      RenderState.SetDepthStencilState(TStaticDepthStencilState<true, CF_DepthNearOrEqual>::GetRHI());
    }

    virtual void AddMeshBatch(
        const FMeshBatch &RESTRICT MeshBatch,
        uint64 BatchElementMask,
        const FPrimitiveSceneProxy *RESTRICT PrimitiveSceneProxy,
        int32 StaticMeshId = -1) override final
    {
      // No fallback material: a fallback would evaluate a different emissive
      // and mask, so a lamp without compiled pass shaders is reported instead.
      const FMaterialRenderProxy *MaterialRenderProxy = MeshBatch.MaterialRenderProxy;
      const FMaterial *Material =
          MaterialRenderProxy ? MaterialRenderProxy->GetMaterialNoFallback(FeatureLevel) : nullptr;
      if (Material == nullptr)
      {
        ++Missing;
        return;
      }
      FMaterialShaderTypes ShaderTypes;
      ShaderTypes.AddShaderType<FCueSimLampVS>();
      ShaderTypes.AddShaderType<FCueSimLampPS>();
      FMaterialShaders Shaders;
      if (!Material->TryGetShaders(ShaderTypes, MeshBatch.VertexFactory->GetType(), Shaders))
      {
        ++Missing;
        return;
      }
      TMeshProcessorShaders<FCueSimLampVS, FCueSimLampPS> PassShaders;
      Shaders.TryGetVertexShader(PassShaders.VertexShader);
      Shaders.TryGetPixelShader(PassShaders.PixelShader);

      const FMeshDrawingPolicyOverrideSettings Override = ComputeMeshOverrideSettings(MeshBatch);
      const ERasterizerFillMode FillMode = ComputeMeshFillMode(*Material, Override);
      // Same face culling as the base pass, so back faces never win.
      const ERasterizerCullMode CullMode = ComputeMeshCullMode(*Material, Override);

      FCueSimLampElementData Data = Element;
      Data.InitializeMeshMaterialData(ViewIfDynamicMeshCommand, PrimitiveSceneProxy, MeshBatch, -1, false);
      const FMeshDrawCommandSortKey SortKey =
          CalculateMeshStaticSortKey(PassShaders.VertexShader, PassShaders.PixelShader);
      BuildMeshDrawCommands(MeshBatch, BatchElementMask, PrimitiveSceneProxy, *MaterialRenderProxy,
          *Material, RenderState, PassShaders, FillMode, CullMode, SortKey,
          EMeshPassFeatures::Default, Data);
      ++Drawn;
    }

    FMeshPassProcessorRenderState RenderState;
    FCueSimLampElementData Element;
    int32 Drawn = 0;
    int32 Missing = 0;
  };

  FVector4f Row(const FMatrix &M, int32 Column)
  {
    // UE matrices act on row vectors: out[c] = sum_r in[r] * M[r][c].
    return FVector4f(M.M[0][Column], M.M[1][Column], M.M[2][Column], M.M[3][Column]);
  }

  TArray<TSharedPtr<FJsonValue>> MatrixJson(const FMatrix &M)
  {
    TArray<TSharedPtr<FJsonValue>> Rows;
    for (int32 R = 0; R < 4; ++R)
    {
      TArray<TSharedPtr<FJsonValue>> Values;
      for (int32 C = 0; C < 4; ++C)
        Values.Add(MakeShared<FJsonValueNumber>(M.M[R][C]));
      Rows.Add(MakeShared<FJsonValueArray>(Values));
    }
    return Rows;
  }

  TArray<TSharedPtr<FJsonValue>> IntsJson(std::initializer_list<int64> Values)
  {
    TArray<TSharedPtr<FJsonValue>> Out;
    for (int64 V : Values)
      Out.Add(MakeShared<FJsonValueNumber>(static_cast<double>(V)));
    return Out;
  }

  FLODMask SelectLOD(const FSceneView &View, const FPrimitiveSceneProxy &Proxy)
  {
    // Mirrors ComputePrimitiveMeshLODs in SceneVisibility.cpp without the GPU
    // scene culling query, which static mesh components do not use.
    const FDesiredLODLevel Desired = Proxy.GetDesiredLODLevel_RenderThread(&View);
    FLODMask Mask;
    if (Desired.IsFixed())
    {
      Mask.SetLOD(Desired.LOD);
      return Mask;
    }
    static const auto *DistanceScale =
        IConsoleManager::Get().FindConsoleVariable(TEXT("r.StaticMeshLODDistanceScale"));
    const float Scale = (DistanceScale ? DistanceScale->GetFloat() : 1.0f) * View.LODDistanceFactor;
    const int32 Forced = View.Family->EngineShowFlags.LOD ? GetCVarForceLOD() : 0;
    const FBoxSphereBounds &Bounds = Proxy.GetBounds();
    float ScreenSizeSquared = 0.0f;
    return ComputeLODForMeshes(Proxy.GetPrimitiveSceneInfo()->StaticMeshRelevances, View,
        Bounds.Origin, Bounds.SphereRadius, Forced, ScreenSizeSquared, Desired.LOD, Scale);
  }

  struct FCueSimLampReadback
  {
    TUniquePtr<FRHIGPUBufferReadback> Counter;
    TUniquePtr<FRHIGPUBufferReadback> Records;
    uint32 MaxRecords = 0;
    FIntRect ViewRect;
  };

  constexpr uint32 CounterWords = 8;
}

// =============================================================================
// -- FCueSimLampPassExtension -------------------------------------------------
// =============================================================================

FCueSimLampPassExtension::FCueSimLampPassExtension(ACueSimLampPassSensor &InSensor)
  : Sensor(&InSensor) {}

void FCueSimLampPassExtension::Detach()
{
  bDetached = true;
}

bool FCueSimLampPassExtension::IsActiveThisFrame(const FSceneViewExtensionContext &Context) const
{
  ACueSimLampPassSensor *Owner = Sensor.Get();
  return !bDetached && Owner != nullptr && Owner->IsPassActive();
}

void FCueSimLampPassExtension::SetupViewFamily(FSceneViewFamily &InViewFamily)
{
  // Inside the camera's CaptureScene(), after SendAllEndOfFrameUpdates: the
  // scene proxies read here are the ones this capture renders.
  ACueSimLampPassSensor *Owner = Sensor.Get();
  if (Owner == nullptr)
    return;
  TSharedPtr<FCueSimLampRequest> Request = Owner->MakeRequest();
  if (InViewFamily.Views.Num() != 1)
    Request->Errors.Add(TEXT("multiple_views"));
  auto Self = AsShared();
  ENQUEUE_RENDER_COMMAND(CueSimLampPassQueue)(
      [Self, Request](FRHICommandListImmediate &) { Self->Pending.Add(Request); });
}

void FCueSimLampPassExtension::PreRenderViewFamily_RenderThread(
    FRDGBuilder &GraphBuilder, FSceneViewFamily &InViewFamily)
{
  // Requests are queued in capture order ahead of the capture's render command.
  Current.Reset();
  bCurrentRendered = false;
  if (!Pending.IsEmpty())
  {
    Current = Pending[0];
    Pending.RemoveAt(0);
  }
}

void FCueSimLampPassExtension::PostRenderViewFamily_RenderThread(
    FRDGBuilder &GraphBuilder, FSceneViewFamily &InViewFamily)
{
  if (Current.IsValid() && !bCurrentRendered)
  {
    Current->Errors.Add(TEXT("base_pass_not_observed"));
    Deliver(Current, {});
  }
  Current.Reset();
}

void FCueSimLampPassExtension::PostRenderBasePassDeferred_RenderThread(
    FRDGBuilder &GraphBuilder,
    FSceneView &View,
    const FRenderTargetBindingSlots &RenderTargets,
    TRDGUniformBufferRef<FSceneTextureUniformParameters> SceneTextures)
{
  if (!Current.IsValid() || bCurrentRendered)
    return;
  bCurrentRendered = true;
  TSharedPtr<FCueSimLampRequest> Request = Current;
  FJsonObject &Header = *Request->Header;

  const FIntRect Rect = View.UnscaledViewRect;
  const FViewMatrices &Matrices = View.ViewMatrices;
  Header.SetArrayField(TEXT("view_rect"), IntsJson({Rect.Min.X, Rect.Min.Y, Rect.Max.X, Rect.Max.Y}));
  Header.SetNumberField(TEXT("width"), Rect.Width());
  Header.SetNumberField(TEXT("height"), Rect.Height());
  Header.SetArrayField(TEXT("projection_matrix"), MatrixJson(Matrices.GetProjectionNoAAMatrix()));
  const FVector2D Jitter = Matrices.GetTemporalAAJitter();
  TArray<TSharedPtr<FJsonValue>> JitterJson = {
      MakeShared<FJsonValueNumber>(Jitter.X), MakeShared<FJsonValueNumber>(Jitter.Y)};
  Header.SetArrayField(TEXT("temporal_aa_jitter_ndc"), JitterJson);
  Header.SetNumberField(TEXT("anti_aliasing_method"), static_cast<int32>(View.AntiAliasingMethod));

  FRDGTextureRef SceneDepth = RenderTargets.DepthStencil.GetTexture();
  if (SceneDepth == nullptr)
    Request->Errors.Add(TEXT("no_scene_depth"));
  else if (Rect.Max.X > SceneDepth->Desc.Extent.X || Rect.Max.Y > SceneDepth->Desc.Extent.Y)
    Request->Errors.Add(TEXT("view_rect_outside_scene_textures"));
  const FScene *Scene = View.Family->Scene ? View.Family->Scene->GetRenderScene() : nullptr;
  if (Scene == nullptr)
    Request->Errors.Add(TEXT("no_render_scene"));
  if (!Request->Errors.IsEmpty() || Rect.Area() <= 0)
  {
    Deliver(Request, {});
    return;
  }

  RDG_EVENT_SCOPE(GraphBuilder, "CueSimLampPass");
  const FIntPoint Extent = SceneDepth->Desc.Extent;
  FRDGTextureRef Ids = GraphBuilder.CreateTexture(FRDGTextureDesc::Create2D(Extent,
      PF_R32G32B32A32_UINT, FClearValueBinding::Transparent,
      TexCreate_RenderTargetable | TexCreate_ShaderResource), TEXT("CueSimLampIds"));
  FRDGTextureRef Emissive = GraphBuilder.CreateTexture(FRDGTextureDesc::Create2D(Extent,
      PF_A32B32G32R32F, FClearValueBinding::Transparent,
      TexCreate_RenderTargetable | TexCreate_ShaderResource), TEXT("CueSimLampEmissive"));
  FRDGTextureRef Local = GraphBuilder.CreateTexture(FRDGTextureDesc::Create2D(Extent,
      PF_A32B32G32R32F, FClearValueBinding::Transparent,
      TexCreate_RenderTargetable | TexCreate_ShaderResource), TEXT("CueSimLampLocal"));
  FRDGTextureRef LampDepth = GraphBuilder.CreateTexture(FRDGTextureDesc::Create2D(Extent,
      PF_DepthStencil, FClearValueBinding::DepthFar,
      TexCreate_DepthStencilTargetable | TexCreate_ShaderResource), TEXT("CueSimLampDepth"));

  // Per-draw constants and the draw status reported in the header.
  struct FPreparedDraw
  {
    const FPrimitiveSceneProxy *Proxy;
    FCueSimLampElementData Element;
    TArray<FMeshBatch> Batches;
    TSharedPtr<FJsonObject> Status;
  };
  TArray<FPreparedDraw> Prepared;
  const FVector PreViewTranslation = Matrices.GetPreViewTranslation();
  const FVector ViewOrigin = Matrices.GetViewOrigin();
  FRHITexture *Mask = Request->MaskResource ? Request->MaskResource->TextureRHI.GetReference() : nullptr;
  for (const FCueSimLampDraw &Draw : Request->Draws)
  {
    FPreparedDraw &Item = Prepared.AddDefaulted_GetRef();
    Item.Proxy = Draw.Proxy;
    Item.Status = MakeShared<FJsonObject>();
    Item.Status->SetNumberField(TEXT("actor_id"), Draw.ActorId);
    Item.Status->SetNumberField(TEXT("component_index"), Draw.ComponentIndex);
    FString Status = TEXT("ok");
    const FPrimitiveSceneProxy &Proxy = *Draw.Proxy;
    const float Distance = FVector::Dist(ViewOrigin, Proxy.GetBounds().Origin);
    if (Mask == nullptr)
      Status = TEXT("mask_texture_unavailable");
    else if (!Proxy.IsShown(&View))
      Status = TEXT("hidden_in_view");
    else if (Proxy.IsNaniteMesh())
      Status = TEXT("nanite_unsupported");
    else if ((Proxy.GetMaxDrawDistance() > 0.0f && Distance > Proxy.GetMaxDrawDistance()) ||
             Distance < Proxy.GetMinDrawDistance())
      Status = TEXT("draw_distance_culled");
    const FLODMask LOD = Status == TEXT("ok") ? SelectLOD(View, Proxy) : FLODMask();
    if (Status == TEXT("ok") && (!LOD.IsValid() || int32(LOD.LODIndex0) >= Draw.LampSectionsPerLOD.Num()))
      Status = TEXT("no_lod");
    if (Status == TEXT("ok"))
    {
      Item.Status->SetNumberField(TEXT("lod"), LOD.LODIndex0);
      // During a dithered LOD transition two LODs share the pixels; the
      // drawn one may not be the depth winner, so the consumer marks it.
      Item.Status->SetBoolField(TEXT("lod_transition"), LOD.bIsRange != 0);
      TArray<FMeshBatch> Batches;
      Proxy.GetMeshDescription(LOD.LODIndex0, Batches);
      const uint64 Sections = Draw.LampSectionsPerLOD[LOD.LODIndex0];
      for (FMeshBatch &Batch : Batches)
      {
        if (Batch.SegmentIndex < 64 && (Sections & (uint64(1) << Batch.SegmentIndex)) != 0)
          Item.Batches.Add(Batch);
      }
      Item.Status->SetNumberField(TEXT("sections"), Item.Batches.Num());
      if (Item.Batches.IsEmpty())
        Status = TEXT("no_lamp_sections");
      const FMatrix ToLocal =
          FTranslationMatrix(-PreViewTranslation) * Proxy.GetLocalToWorld().Inverse();
      Item.Element.ActorId = Draw.ActorId;
      Item.Element.ToLocal0 = Row(ToLocal, 0);
      Item.Element.ToLocal1 = Row(ToLocal, 1);
      Item.Element.ToLocal2 = Row(ToLocal, 2);
      Item.Element.Mask = Mask;
    }
    Item.Status->SetStringField(TEXT("status"), Status);
    if (Status != TEXT("ok"))
      Item.Batches.Reset();
  }

  auto *MeshParameters = GraphBuilder.AllocParameters<FCueSimLampMeshPassParameters>();
  MeshParameters->View = View.ViewUniformBuffer;
  MeshParameters->Scene = GetSceneUniformBufferRef(GraphBuilder, View);
  MeshParameters->RenderTargets[0] = FRenderTargetBinding(Ids, ERenderTargetLoadAction::EClear);
  MeshParameters->RenderTargets[1] = FRenderTargetBinding(Emissive, ERenderTargetLoadAction::EClear);
  MeshParameters->RenderTargets[2] = FRenderTargetBinding(Local, ERenderTargetLoadAction::EClear);
  MeshParameters->RenderTargets.DepthStencil = FDepthStencilBinding(LampDepth,
      ERenderTargetLoadAction::EClear, ERenderTargetLoadAction::ENoAction,
      FExclusiveDepthStencil::DepthWrite_StencilNop);
  AddSimpleMeshPass(GraphBuilder, MeshParameters, Scene, View, nullptr,
      RDG_EVENT_NAME("CueSimLampMeshes"), Rect,
      [Scene, &View, &Prepared](FDynamicPassMeshDrawListContext *Context)
      {
        // The material needs this scene's parameter collections, including VisualTime.
        FCueSimLampMeshProcessor Processor(Scene, &View, Context);
        for (FPreparedDraw &Item : Prepared)
        {
          Processor.Element = Item.Element;
          const int32 Missing = Processor.Missing;
          for (const FMeshBatch &Batch : Item.Batches)
          {
            Processor.AddMeshBatch(Batch, ~0ull, Item.Proxy);
          }
          if (Processor.Missing != Missing)
            Item.Status->SetStringField(TEXT("status"), TEXT("lamp_shader_missing"));
        }
      });
  TArray<TSharedPtr<FJsonValue>> DrawJson;
  for (const FPreparedDraw &Item : Prepared)
    DrawJson.Add(MakeShared<FJsonValueObject>(Item.Status));
  Header.SetArrayField(TEXT("draws"), DrawJson);

  TArray<FCueSimLampBoxGPU> Boxes;
  for (const FCueSimLampBox &Box : Request->Boxes)
  {
    const FMatrix ToBox = FTranslationMatrix(-PreViewTranslation) * Box.WorldToBox;
    FCueSimLampBoxGPU &Out = Boxes.AddDefaulted_GetRef();
    Out.Row0 = Row(ToBox, 0);
    Out.Row1 = Row(ToBox, 1);
    Out.Row2 = Row(ToBox, 2);
    Out.ExtentId = FVector4f(FVector3f(Box.Extent), FMath::AsFloat(Box.ActorId));
  }
  const uint32 MaxRecords = Request->MaxRecords;
  FRDGBufferRef Records = GraphBuilder.CreateBuffer(
      FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), MaxRecords * CueSimLampPass::RecordWords),
      TEXT("CueSimLampRecords"));
  FRDGBufferRef Counter = GraphBuilder.CreateBuffer(
      FRDGBufferDesc::CreateStructuredDesc(sizeof(uint32), CounterWords), TEXT("CueSimLampCounter"));
  FRDGBufferUAVRef CounterUAV = GraphBuilder.CreateUAV(Counter);
  AddClearUAVPass(GraphBuilder, CounterUAV, 0u);

  auto *Classify = GraphBuilder.AllocParameters<FCueSimLampClassifyCS::FParameters>();
  Classify->View = View.ViewUniformBuffer;
  Classify->CueSimLampIds = Ids;
  Classify->CueSimLampEmissive = Emissive;
  Classify->CueSimLampLocal = Local;
  Classify->CueSimSceneDepth = SceneDepth;
  Classify->CueSimBoxes = GraphBuilder.CreateSRV(
      CreateStructuredBuffer(GraphBuilder, TEXT("CueSimLampBoxes"), Boxes));
  Classify->CueSimNumBoxes = Boxes.Num();
  Classify->CueSimViewMin = Rect.Min;
  Classify->CueSimViewSize = Rect.Size();
  Classify->CueSimDepthToleranceCm = Request->DepthToleranceCm;
  Classify->CueSimDepthToleranceRelative = Request->DepthToleranceRelative;
  Classify->CueSimEmittingThreshold = Request->EmittingThreshold;
  Classify->CueSimMaxRecords = MaxRecords;
  Classify->CueSimRecords = GraphBuilder.CreateUAV(Records);
  Classify->CueSimCounter = CounterUAV;
  TShaderMapRef<FCueSimLampClassifyCS> Shader(GetGlobalShaderMap(View.GetFeatureLevel()));
  FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("CueSimLampClassify"), Shader, Classify,
      FComputeShaderUtils::GetGroupCount(Rect.Size(), FIntPoint(8, 8)));

  auto Readback = MakeShared<FCueSimLampReadback, ESPMode::ThreadSafe>();
  Readback->Counter = MakeUnique<FRHIGPUBufferReadback>(TEXT("CueSimLampCounter"));
  Readback->Records = MakeUnique<FRHIGPUBufferReadback>(TEXT("CueSimLampRecords"));
  Readback->MaxRecords = MaxRecords;
  Readback->ViewRect = Rect;
  AddEnqueueCopyPass(GraphBuilder, Readback->Counter.Get(), Counter, CounterWords * sizeof(uint32));
  AddEnqueueCopyPass(GraphBuilder, Readback->Records.Get(), Records,
      MaxRecords * CueSimLampPass::RecordWords * sizeof(uint32));

  auto Self = AsShared();
  GraphBuilder.AddPostExecuteCallback([Self, Request, Readback]()
  {
    // Poll on a background worker, never on a named thread (see ImageUtil.cpp),
    // then map on the render thread once the copies have completed.
    AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [Self, Request, Readback]()
    {
      uint64 Spins = 0;
      while (!(Readback->Counter->IsReady() && Readback->Records->IsReady()))
      {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        if (++Spins > 25000u)  // ~5 s
        {
          Request->Errors.Add(TEXT("readback_timeout"));
          Self->Deliver(Request, {});
          return;
        }
      }
      ENQUEUE_RENDER_COMMAND(CueSimLampPassMap)([Self, Request, Readback](FRHICommandListImmediate &)
      {
        uint32 Words[CounterWords];
        FMemory::Memcpy(Words, Readback->Counter->Lock(sizeof(Words)), sizeof(Words));
        Readback->Counter->Unlock();
        const uint32 Count = FMath::Min(Words[0], Readback->MaxRecords);
        if (Words[0] > Readback->MaxRecords)
          Request->Errors.Add(TEXT("record_overflow"));
        const FIntRect &Rect = Readback->ViewRect;
        if (Words[1] != uint32(Rect.Min.X) || Words[2] != uint32(Rect.Min.Y) ||
            Words[3] != uint32(Rect.Width()) || Words[4] != uint32(Rect.Height()))
          Request->Errors.Add(TEXT("view_rect_mismatch"));
        TArray<uint32> Records;
        if (Count > 0)
        {
          const uint32 Bytes = Count * CueSimLampPass::RecordWords * sizeof(uint32);
          Records.SetNumUninitialized(static_cast<int32>(Count * CueSimLampPass::RecordWords));
          FMemory::Memcpy(Records.GetData(), Readback->Records->Lock(Bytes), Bytes);
          Readback->Records->Unlock();
        }
        Self->Deliver(Request, MoveTemp(Records));
      });
    });
  });
}

void FCueSimLampPassExtension::Deliver(TSharedPtr<FCueSimLampRequest> Request, TArray<uint32> Records)
{
  auto Self = AsShared();
  AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask,
      [Self, Request, Records = MoveTemp(Records)]() mutable
  {
    ACueSimLampPassSensor *Owner = Self->Sensor.Get();
    if (Self->bDetached || Owner == nullptr)
      return;
    // Atomic appends arrive in GPU order; sort by pixel for a deterministic packet.
    using Record = TStaticArray<uint32, CueSimLampPass::RecordWords>;
    static_assert(sizeof(Record) == CueSimLampPass::RecordWords * sizeof(uint32));
    Record *Begin = reinterpret_cast<Record *>(Records.GetData());
    const int32 Count = Records.Num() / static_cast<int32>(CueSimLampPass::RecordWords);
    std::sort(Begin, Begin + Count, [](const Record &A, const Record &B) { return A[0] < B[0]; });

    FJsonObject &Header = *Request->Header;
    TArray<TSharedPtr<FJsonValue>> Errors;
    for (const FString &Error : Request->Errors)
      Errors.Add(MakeShared<FJsonValueString>(Error));
    Header.SetStringField(TEXT("status"), Request->Errors.IsEmpty() ? TEXT("ok") : TEXT("error"));
    Header.SetArrayField(TEXT("errors"), Errors);
    Header.SetNumberField(TEXT("record_count"), Count);
    FString Json;
    auto Writer = TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Json);
    FJsonSerializer::Serialize(Request->Header.ToSharedRef(), Writer);
    FTCHARToUTF8 Utf8(*Json);

    // "CSLP", version, header bytes, record count, header (padded to 4), records.
    TArray<uint8> Packet;
    const uint32 HeaderBytes = Utf8.Length();
    const uint32 Padded = Align(HeaderBytes, 4u);
    const uint32 Fixed[4] = {0x504C5343u, 1u, HeaderBytes, uint32(Count)};
    Packet.Append(reinterpret_cast<const uint8 *>(Fixed), static_cast<int32>(sizeof(Fixed)));
    Packet.Append(reinterpret_cast<const uint8 *>(Utf8.Get()), static_cast<int32>(HeaderBytes));
    Packet.AddZeroed(static_cast<int32>(Padded - HeaderBytes));
    Packet.Append(reinterpret_cast<const uint8 *>(Records.GetData()), Records.Num() * static_cast<int32>(sizeof(uint32)));
    Owner->SendPacket(MoveTemp(Packet), Request->FrameIndex, Request->Timestamp, Request->CameraTransform);
  });
}
