// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "Carla/CueSim/CueSimLampModel.h"

#include <util/ue-header-guard-begin.h>
#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include <util/ue-header-guard-end.h>

#include "CueSimLampSubsystem.generated.h"

class ACarlaWheeledVehicle;
class ULightComponent;
class UMaterialInstanceDynamic;
class UMaterialParameterCollection;

/// CueSim lamp controller: owns the lamp timeline of vehicles given a schedule.
///
/// Off unless `carla.CueSim.LampController` is 1. A controlled vehicle ignores the
/// stock light-state path (`ACarlaWheeledVehicle::SetCueSimLampControlled`); each
/// world tick, after timers and before the sensors capture in OnWorldPostActorTick,
/// this subsystem evaluates the schedule at the frame's simulation time
/// (`CueSimLamp::Evaluate`), writes every lamp group's material scalar and real light
/// components from the exposure-integrated value, reads the material back and logs it.
/// While any vehicle is controlled, the material blink gate (`VisualTime` in the
/// CarlaParameters collection) is pinned to `carla.CueSim.LampController.VisualTimePin`.
UCLASS()
class CARLA_API UCueSimLampSubsystem : public UTickableWorldSubsystem
{
  GENERATED_BODY()

public:

  static bool IsEnabled();

  /// Parses and applies a `cuesim-lamp-schedule-v1` payload; an empty payload
  /// releases the vehicle. Returns an error message, empty on success.
  FString SetSchedule(ACarlaWheeledVehicle &Vehicle, uint32 ActorId, const FString &Payload);

  /// Log rows recorded for this actor since the previous call, one JSON object per
  /// line. Fails, and drops the rows, if the buffer overflowed.
  bool TakeLog(uint32 ActorId, FString &OutJsonLines, FString &OutError);

  virtual void Tick(float DeltaTime) override;

  virtual TStatId GetStatId() const override;

  virtual void Deinitialize() override;

protected:

  virtual bool DoesSupportWorldType(const EWorldType::Type WorldType) const override;

private:

  static constexpr int32 FunctionCount = 8;

  struct FControlled
  {
    TWeakObjectPtr<ACarlaWheeledVehicle> Vehicle;
    FString ScheduleId;
    CueSimLamp::FExposure Exposure;
    CueSimLamp::FFunction Functions[FunctionCount];
    double MaterialIntensity = 0.0;
    TArray<TWeakObjectPtr<UMaterialInstanceDynamic>> Materials;
    bool bParameterFound[FunctionCount] = {};
    TArray<TWeakObjectPtr<ULightComponent>> Lights[FunctionCount];
    TArray<float> LightBaseIntensity[FunctionCount];
  };

  void Release(FControlled &Entry);

  void ReleaseAll();

  /// Pins the blink gate and returns the collection value read back, or NaN.
  float PinVisualTime();

  void Apply(uint32 ActorId, FControlled &Entry, uint64 Frame, double SimTime, float VisualTime);

  TMap<uint32, FControlled> Controlled;

  TMap<uint32, TArray<FString>> Logs;

  TSet<uint32> Overflowed;

  UPROPERTY(Transient)
  TObjectPtr<UMaterialParameterCollection> CarlaParameters = nullptr;
};
