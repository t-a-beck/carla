// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

// CueSim lamp pass: per-pixel lamp identity, visibility and evaluated lamp
// emission in the exact view of an RGB camera, without a second scene render
// and without Unreal Engine changes.
//
// A scene view extension is attached to the RGB camera's own capture
// component (USceneCaptureComponent2D::SceneViewExtensions), so it only runs
// inside that camera's view family. After that view's base pass it rasterises
// the registered lamp mesh sections once more with the lamp's own material
// (same view uniform buffer, jitter, LOD and material instance values) into
// auxiliary targets, compares the nearest lamp fragment with the view's final
// opaque depth, and appends one record per lamp pixel. Scene colour, depth,
// GBuffers and every later pass are untouched, so the RGB image is identical
// with the pass on or off. The wire format is documented in
// docker/sim/renderer-lamps/README.md of the AutoCue repository.

#include "CoreMinimal.h"
#include "SceneViewExtension.h"

#include <atomic>

class ACueSimLampPassSensor;
class ASceneCaptureSensor;
class FPrimitiveSceneProxy;
class FTextureResource;
class FJsonObject;
struct FSensorCaptureContext;

namespace CueSimLampPass
{
  /// Packet schema written into the JSON header.
  inline constexpr const TCHAR *Schema = TEXT("cuesim-lamp-pass-v1");

  /// 32-bit words per record; the layout is fixed by the shader and the client.
  inline constexpr uint32 RecordWords = 12;

  enum class EVisibility : uint32
  {
    Visible = 1,
    OtherBlocked = 2,
    SelfBlocked = 3,
    Unresolved = 4,
  };

  enum ERecordFlags : uint32
  {
    Emitting = 1u << 0,
    NonFiniteEmissive = 1u << 1,
    NoTexCoord = 1u << 2,
    OccluderAmbiguous = 1u << 3,
  };
}

/// One registered lamp mesh component of one actor, resolved on the game thread
/// right before the camera's capture is submitted.
struct FCueSimLampDraw
{
  uint32 ActorId = 0;
  int32 ComponentIndex = 0;
  const FPrimitiveSceneProxy *Proxy = nullptr;
  /// Bit S of entry L: section S of LOD L uses the lamp material.
  TArray<uint64> LampSectionsPerLOD;
};

/// Oriented box of one actor, used to attribute an occluding depth sample.
struct FCueSimLampBox
{
  uint32 ActorId = 0;
  FMatrix WorldToBox = FMatrix::Identity;
  FVector Extent = FVector::ZeroVector;
};

/// Everything one capture of the camera needs, created on the game thread.
struct FCueSimLampRequest
{
  uint64 FrameIndex = 0;
  double Timestamp = 0.0;
  uint64 CaptureSequence = 0;
  FTransform CameraTransform;
  TArray<FCueSimLampDraw> Draws;
  TArray<FCueSimLampBox> Boxes;
  FTextureResource *MaskResource = nullptr;
  float EmittingThreshold = 0.0f;
  float DepthToleranceCm = 0.5f;
  float DepthToleranceRelative = 1e-4f;
  uint32 MaxRecords = 0;
  /// Header fields known on the game thread (camera, actors, settings).
  TSharedPtr<FJsonObject> Header;
  /// Non-empty when this capture cannot produce trustworthy records.
  TArray<FString> Errors;
};

/// Scene view extension registered only on one camera's capture component.
class FCueSimLampPassExtension
  : public ISceneViewExtension,
    public TSharedFromThis<FCueSimLampPassExtension, ESPMode::ThreadSafe>
{
public:
  explicit FCueSimLampPassExtension(ACueSimLampPassSensor &InSensor);

  // Game thread.
  virtual void SetupViewFamily(FSceneViewFamily &InViewFamily) override;
  virtual bool IsActiveThisFrame(const FSceneViewExtensionContext &Context) const override;

  // Render thread.
  virtual void PreRenderViewFamily_RenderThread(
      FRDGBuilder &GraphBuilder, FSceneViewFamily &InViewFamily) override;
  virtual void PostRenderBasePassDeferred_RenderThread(
      FRDGBuilder &GraphBuilder,
      FSceneView &InView,
      const FRenderTargetBindingSlots &RenderTargets,
      TRDGUniformBufferRef<FSceneTextureUniformParameters> SceneTextures) override;
  virtual void PostRenderViewFamily_RenderThread(
      FRDGBuilder &GraphBuilder, FSceneViewFamily &InViewFamily) override;

  /// Game thread: called by the sensor before it is destroyed.
  void Detach();

private:
  void Deliver(TSharedPtr<FCueSimLampRequest> Request, TArray<uint32> Records);

  TWeakObjectPtr<ACueSimLampPassSensor> Sensor;
  std::atomic<bool> bDetached{false};

  // Render-thread state: requests in capture order, and the one being rendered.
  TArray<TSharedPtr<FCueSimLampRequest>> Pending;
  TSharedPtr<FCueSimLampRequest> Current;
  bool bCurrentRendered = false;
};
