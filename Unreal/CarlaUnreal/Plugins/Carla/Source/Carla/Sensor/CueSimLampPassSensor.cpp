// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla/Sensor/CueSimLampPassSensor.h"
#include "Carla.h"
#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"
#include "Carla/Actor/ActorRegistry.h"
#include "Carla/Actor/CarlaActor.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Carla/Sensor/CueSimLampPass.h"
#include "Carla/Sensor/SceneCaptureCamera.h"
#include "Carla/Util/BoundingBoxCalculator.h"

#include <util/ue-header-guard-begin.h>
#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture2D.h"
#include "HAL/IConsoleManager.h"
#include "Materials/Material.h"
#include "Materials/MaterialInterface.h"
#include "StaticMeshResources.h"
#include <util/ue-header-guard-end.h>

static TAutoConsoleVariable<int32> CVarCueSimLampPass(
    TEXT("carla.CueSim.LampPass"),
    1,
    TEXT("Run the CueSim same-view lamp pass for spawned sensor.other.cuesim_lamp_pass ")
    TEXT("sensors (0 disables it; the RGB path is identical either way)."),
    ECVF_Default);

namespace
{
  TSharedPtr<FJsonObject> TransformJson(const FTransform &Transform)
  {
    // carla.Transform convention: metres, degrees.
    const FVector L = Transform.GetLocation() / 100.0;
    const FRotator R = Transform.Rotator();
    auto Location = MakeShared<FJsonObject>();
    Location->SetNumberField(TEXT("x"), L.X);
    Location->SetNumberField(TEXT("y"), L.Y);
    Location->SetNumberField(TEXT("z"), L.Z);
    auto Rotation = MakeShared<FJsonObject>();
    Rotation->SetNumberField(TEXT("pitch"), R.Pitch);
    Rotation->SetNumberField(TEXT("yaw"), R.Yaw);
    Rotation->SetNumberField(TEXT("roll"), R.Roll);
    auto Out = MakeShared<FJsonObject>();
    Out->SetObjectField(TEXT("location"), Location);
    Out->SetObjectField(TEXT("rotation"), Rotation);
    return Out;
  }

  bool IsLampSlot(UMaterialInterface *Material, const FString &LampMaterial)
  {
    UMaterial *Base = Material != nullptr ? Material->GetBaseMaterial() : nullptr;
    return Base != nullptr && Base->GetName().Contains(LampMaterial);
  }
}

ACueSimLampPassSensor::ACueSimLampPassSensor(const FObjectInitializer &ObjectInitializer)
  : Super(ObjectInitializer)
{
  PrimaryActorTick.bCanEverTick = true;
}

FActorDefinition ACueSimLampPassSensor::GetSensorDefinition()
{
  auto Definition = UActorBlueprintFunctionLibrary::MakeGenericSensorDefinition(
      TEXT("other"), TEXT("cuesim_lamp_pass"));
  auto Add = [&Definition](const TCHAR *Id, EActorAttributeType Type, const FString &Value)
  {
    FActorVariation Variation;
    Variation.Id = Id;
    Variation.Type = Type;
    Variation.RecommendedValues = {Value};
    Variation.bRestrictToRecommended = false;
    Definition.Variations.Emplace(Variation);
  };
  Add(TEXT("mask_texture"), EActorAttributeType::String,
      TEXT("/Game/Carla/Static/GenericMaterials/000_Masters/Textures/T_8Color_msk.T_8Color_msk"));
  Add(TEXT("lamp_material"), EActorAttributeType::String, TEXT("VehicleLightsMaster"));
  Add(TEXT("emitting_threshold"), EActorAttributeType::Float, TEXT("0.0"));
  Add(TEXT("depth_tolerance_cm"), EActorAttributeType::Float, TEXT("0.5"));
  Add(TEXT("depth_tolerance_relative"), EActorAttributeType::Float, TEXT("0.0001"));
  Add(TEXT("box_margin_cm"), EActorAttributeType::Float, TEXT("10.0"));
  Add(TEXT("max_records"), EActorAttributeType::Int, TEXT("131072"));
  return Definition;
}

void ACueSimLampPassSensor::Set(const FActorDescription &Description)
{
  Super::Set(Description);
  const auto &V = Description.Variations;
  MaskTexturePath = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToString(
      TEXT("mask_texture"), V, MaskTexturePath);
  LampMaterial = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToString(
      TEXT("lamp_material"), V, LampMaterial);
  EmittingThreshold = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
      TEXT("emitting_threshold"), V, EmittingThreshold);
  DepthToleranceCm = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
      TEXT("depth_tolerance_cm"), V, DepthToleranceCm);
  DepthToleranceRelative = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
      TEXT("depth_tolerance_relative"), V, DepthToleranceRelative);
  BoxMarginCm = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
      TEXT("box_margin_cm"), V, BoxMarginCm);
  MaxRecords = FMath::Clamp(UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt(
      TEXT("max_records"), V, MaxRecords), 1, 1 << 22);
  MaskTexture = LoadObject<UTexture2D>(nullptr, *MaskTexturePath);
  if (MaskTexture == nullptr)
  {
    UE_LOG(LogCarla, Error, TEXT("CueSim lamp pass: mask texture %s not found"), *MaskTexturePath);
  }
}

void ACueSimLampPassSensor::BindToCamera()
{
  if (Extension.IsValid())
    return;
  auto *Parent = Cast<ASceneCaptureCamera>(GetAttachParentActor());
  if (Parent == nullptr)
    return;  // CARLA attaches after BeginPlay; retried every tick.
  Camera = Parent;
  Extension = MakeShared<FCueSimLampPassExtension, ESPMode::ThreadSafe>(*this);
  Parent->GetCaptureComponent2D()->SceneViewExtensions.Add(Extension);
  UE_LOG(LogCarla, Log, TEXT("CueSim lamp pass bound to camera %s"), *Parent->GetName());
}

void ACueSimLampPassSensor::PrePhysTick(float DeltaSeconds)
{
  Super::PrePhysTick(DeltaSeconds);
  BindToCamera();
}

void ACueSimLampPassSensor::PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaSeconds)
{
  Super::PostPhysTick(World, TickType, DeltaSeconds);
  BindToCamera();
}

void ACueSimLampPassSensor::EndPlay(EEndPlayReason::Type EndPlayReason)
{
  if (Extension.IsValid())
  {
    Extension->Detach();
    if (Camera.IsValid())
    {
      Camera->GetCaptureComponent2D()->SceneViewExtensions.RemoveAll(
          [this](const TWeakPtr<ISceneViewExtension, ESPMode::ThreadSafe> &Item)
          { return Item.Pin().Get() == Extension.Get(); });
    }
    Extension.Reset();
  }
  Super::EndPlay(EndPlayReason);
}

bool ACueSimLampPassSensor::IsPassActive()
{
  return CVarCueSimLampPass.GetValueOnGameThread() != 0 && Camera.IsValid() &&
      AreClientsListening();
}

TSharedPtr<FCueSimLampRequest> ACueSimLampPassSensor::MakeRequest()
{
  auto Request = MakeShared<FCueSimLampRequest>();
  ASceneCaptureCamera *Cam = Camera.Get();
  check(Cam != nullptr);
  // Identical to the context the RGB camera stamps on its image in this tick.
  const FSensorCaptureContext Context = ASensor::MakeCaptureContext(*Cam);
  Request->FrameIndex = Context.FrameIndex;
  Request->Timestamp = Context.Timestamp;
  Request->CameraTransform = Context.Transform;
  Request->CaptureSequence = CaptureSequence++;
  Request->MaskResource = MaskTexture != nullptr ? MaskTexture->GetResource() : nullptr;
  Request->EmittingThreshold = EmittingThreshold;
  Request->DepthToleranceCm = DepthToleranceCm;
  Request->DepthToleranceRelative = DepthToleranceRelative;
  Request->MaxRecords = static_cast<uint32>(MaxRecords);
  if (Request->MaskResource == nullptr)
    Request->Errors.Add(TEXT("mask_texture_unavailable"));

  const UCarlaEpisode &Episode = GetEpisode();
  const FCarlaActor *CameraView = Episode.FindCarlaActor(Cam);
  const FCarlaActor *SelfView = Episode.FindCarlaActor(this);
  const USceneCaptureComponent2D *Capture = Cam->GetCaptureComponent2D();
  if (!FMath::IsNearlyEqual(Capture->CaptureScreenPercentage, 100.0f))
    Request->Errors.Add(TEXT("screen_percentage_unsupported"));

  auto Header = MakeShared<FJsonObject>();
  Header->SetStringField(TEXT("schema"), CueSimLampPass::Schema);
  Header->SetNumberField(TEXT("frame"), static_cast<double>(Request->FrameIndex));
  Header->SetNumberField(TEXT("timestamp"), Request->Timestamp);
  Header->SetNumberField(TEXT("capture_sequence"), static_cast<double>(Request->CaptureSequence));
  Header->SetNumberField(TEXT("camera_actor_id"), CameraView ? CameraView->GetActorId() : 0);
  Header->SetNumberField(TEXT("sensor_actor_id"), SelfView ? SelfView->GetActorId() : 0);
  Header->SetObjectField(TEXT("camera_transform"), TransformJson(Request->CameraTransform));
  Header->SetNumberField(TEXT("fov"), Capture->FOVAngle);
  Header->SetNumberField(TEXT("screen_percentage"), Capture->CaptureScreenPercentage);
  Header->SetStringField(TEXT("lamp_material"), LampMaterial);
  Header->SetStringField(TEXT("mask_texture"), MaskTexturePath);
  Header->SetNumberField(TEXT("emitting_threshold"), EmittingThreshold);
  Header->SetNumberField(TEXT("depth_tolerance_cm"), DepthToleranceCm);
  Header->SetNumberField(TEXT("depth_tolerance_relative"), DepthToleranceRelative);
  Header->SetNumberField(TEXT("box_margin_cm"), BoxMarginCm);
  Header->SetNumberField(TEXT("max_records"), MaxRecords);

  TArray<TSharedPtr<FJsonValue>> Actors;
  for (const auto &Pair : Episode.GetActorRegistry())
  {
    const FCarlaActor *View = Pair.Value.Get();
    if (View == nullptr)
      continue;
    const auto Type = View->GetActorType();
    if (Type != FCarlaActor::ActorType::Vehicle && Type != FCarlaActor::ActorType::Walker)
      continue;
    const AActor *Actor = View->GetActor();
    if (Actor == nullptr)
      continue;  // dormant
    const uint32 ActorId = View->GetActorId();

    // Occluder attribution box for every vehicle and walker.
    const FBoundingBox Box = UBoundingBoxCalculator::GetActorBoundingBox(Actor);
    FCueSimLampBox &Occluder = Request->Boxes.AddDefaulted_GetRef();
    Occluder.ActorId = ActorId;
    const FTransform BoxToWorld = FTransform(Box.Rotation, Box.Origin) * Actor->GetActorTransform();
    Occluder.WorldToBox = BoxToWorld.ToMatrixWithScale().Inverse();
    Occluder.Extent = Box.Extent + FVector(BoxMarginCm);

    if (Type != FCarlaActor::ActorType::Vehicle)
      continue;
    TArray<UStaticMeshComponent *> Components;
    Actor->GetComponents<UStaticMeshComponent>(Components);
    TArray<TSharedPtr<FJsonValue>> ComponentJson;
    for (int32 Index = 0; Index < Components.Num(); ++Index)
    {
      UStaticMeshComponent *Component = Components[Index];
      TArray<int32> Slots;
      for (int32 Slot = 0; Slot < Component->GetNumMaterials(); ++Slot)
      {
        if (IsLampSlot(Component->GetMaterial(Slot), LampMaterial))
          Slots.Add(Slot);
      }
      if (Slots.IsEmpty())
        continue;
      auto Json = MakeShared<FJsonObject>();
      Json->SetNumberField(TEXT("index"), Index);
      Json->SetStringField(TEXT("name"), Component->GetName());
      const UStaticMesh *Mesh = Component->GetStaticMesh();
      Json->SetStringField(TEXT("mesh"), Mesh ? Mesh->GetPathName() : FString());
      TArray<TSharedPtr<FJsonValue>> SlotJson;
      for (int32 Slot : Slots)
      {
        const UMaterialInterface *Material = Component->GetMaterial(Slot);
        auto Entry = MakeShared<FJsonObject>();
        Entry->SetNumberField(TEXT("slot"), Slot);
        Entry->SetStringField(TEXT("material"), Material ? Material->GetPathName() : FString());
        SlotJson.Add(MakeShared<FJsonValueObject>(Entry));
      }
      Json->SetArrayField(TEXT("slots"), SlotJson);
      FString Status = TEXT("ok");
      const FStaticMeshRenderData *RenderData = Mesh ? Mesh->GetRenderData() : nullptr;
      const FPrimitiveSceneProxy *Proxy = Component->GetSceneProxy();
      if (!Component->IsVisible() || Component->bHiddenInGame)
        Status = TEXT("hidden");
      else if (RenderData == nullptr)
        Status = TEXT("no_render_data");
      else if (Proxy == nullptr)
        Status = TEXT("no_scene_proxy");
      else if (Component->HasValidNaniteData())
        Status = TEXT("nanite_unsupported");
      if (Status == TEXT("ok"))
      {
        FCueSimLampDraw &Draw = Request->Draws.AddDefaulted_GetRef();
        Draw.ActorId = ActorId;
        Draw.ComponentIndex = Index;
        Draw.Proxy = Proxy;
        for (const FStaticMeshLODResources &LOD : RenderData->LODResources)
        {
          uint64 Bits = 0;
          for (int32 Section = 0; Section < LOD.Sections.Num() && Section < 64; ++Section)
          {
            if (Slots.Contains(LOD.Sections[Section].MaterialIndex))
              Bits |= uint64(1) << Section;
          }
          Draw.LampSectionsPerLOD.Add(Bits);
        }
        Json->SetNumberField(TEXT("lod_count"), RenderData->LODResources.Num());
      }
      Json->SetStringField(TEXT("status"), Status);
      ComponentJson.Add(MakeShared<FJsonValueObject>(Json));
    }
    if (ComponentJson.IsEmpty())
      continue;
    auto Entry = MakeShared<FJsonObject>();
    Entry->SetNumberField(TEXT("id"), ActorId);
    Entry->SetStringField(TEXT("blueprint"), View->GetActorInfo()->Description.Id);
    Entry->SetObjectField(TEXT("transform"), TransformJson(Actor->GetActorTransform()));
    Entry->SetArrayField(TEXT("components"), ComponentJson);
    Actors.Add(MakeShared<FJsonValueObject>(Entry));
  }
  Header->SetArrayField(TEXT("actors"), Actors);
  Request->Header = Header;
  return Request;
}

void ACueSimLampPassSensor::SendPacket(
    TArray<uint8> &&Packet, uint64 FrameIndex, double Timestamp, const FTransform &CameraTransform)
{
  // ImageSerializer reads the size from the sensor, so packets go out one at a time.
  FScopeLock Lock(&SendLock);
  // ImageSerializer::Deserialize sets alpha to 255 on every pixel.
  // Keep the packet in BGR and leave alpha out of the wire format.
  const int32 PayloadRowBytes = static_cast<int32>(PacketWidth * 3u);
  const int32 Rows = FMath::DivideAndRoundUp(Packet.Num(), PayloadRowBytes);
  PacketHeight = static_cast<uint32>(Rows);
  Packet.SetNumZeroed(Rows * PayloadRowBytes);
  const int32 Pixels = Packet.Num() / 3;
  TArray<uint8> Bitmap;
  Bitmap.SetNumUninitialized(Pixels * 4);
  for (int32 Pixel = 0; Pixel < Pixels; ++Pixel)
  {
    FMemory::Memcpy(Bitmap.GetData() + 4 * Pixel, Packet.GetData() + 3 * Pixel, 3);
    Bitmap[4 * Pixel + 3] = 255;
  }
  FSensorCaptureContext Context;
  Context.FrameIndex = FrameIndex;
  Context.Timestamp = Timestamp;
  Context.Transform = CameraTransform;
  Context.RelativeTransform = FTransform::Identity;
  SendDataToClient(*this, TArrayView<const uint8>(Bitmap), Context);
}
