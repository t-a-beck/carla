// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "Carla/Actor/ActorDefinition.h"
#include "Carla/Sensor/Sensor.h"

#include <util/ue-header-guard-begin.h>
#include "HAL/CriticalSection.h"
#include <util/ue-header-guard-end.h>

#include "CueSimLampPassSensor.generated.h"

class ASceneCaptureCamera;
class FCueSimLampPassExtension;
struct FCueSimLampRequest;
class UTexture2D;

/// sensor.other.cuesim_lamp_pass: attach to a sensor.camera.rgb with an
/// identity transform. Every capture of that camera produces one packet on
/// this sensor's stream (frame number and timestamp equal to the RGB image's),
/// carried as a carla.Image whose raw_data holds the packet bytes. Not
/// spawning the sensor, or setting carla.CueSim.LampPass 0, leaves the camera
/// on its unmodified path.
UCLASS()
class CARLA_API ACueSimLampPassSensor : public ASensor
{
  GENERATED_BODY()

public:

  ACueSimLampPassSensor(const FObjectInitializer &ObjectInitializer);

  static FActorDefinition GetSensorDefinition();

  void Set(const FActorDescription &Description) override;

  virtual void PrePhysTick(float DeltaSeconds) override;
  virtual void PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaSeconds) override;

  /// Game thread. True when the camera's next capture should run the pass.
  bool IsPassActive();

  /// Game thread, from the view extension inside the camera's CaptureScene().
  TSharedPtr<FCueSimLampRequest> MakeRequest();

  /// Any thread. Sends one packet with the camera capture's frame identity.
  void SendPacket(TArray<uint8> &&Packet, uint64 FrameIndex, double Timestamp,
      const FTransform &CameraTransform);

  // carla::sensor::s11n::ImageSerializer transport of the packet bytes.
  uint32 GetImageWidth() const { return PacketWidth; }
  uint32 GetImageHeight() const { return PacketHeight; }
  float GetFOVAngle() const { return 0.0f; }

  /// Width in BGRA texels of the transport image; height varies per packet.
  static constexpr uint32 PacketWidth = 1024u;

protected:

  void EndPlay(EEndPlayReason::Type EndPlayReason) override;

private:

  void BindToCamera();

  TWeakObjectPtr<ASceneCaptureCamera> Camera;
  TSharedPtr<FCueSimLampPassExtension, ESPMode::ThreadSafe> Extension;

  /// Asset path of the lamp mask atlas sampled with the lamp mesh UV0.
  UPROPERTY()
  TObjectPtr<UTexture2D> MaskTexture = nullptr;

  FString MaskTexturePath = TEXT("/Game/Carla/Static/GenericMaterials/000_Masters/Textures/T_8Color_msk.T_8Color_msk");
  /// Substring of the base material name of a lamp material slot.
  FString LampMaterial = TEXT("VehicleLightsMaster");
  float EmittingThreshold = 0.0f;
  float DepthToleranceCm = 0.5f;
  float DepthToleranceRelative = 1e-4f;
  float BoxMarginCm = 10.0f;
  int32 MaxRecords = 131072;

  uint64 CaptureSequence = 0;

  FCriticalSection SendLock;
  uint32 PacketHeight = 0;
};
