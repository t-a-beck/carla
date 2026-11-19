// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla/CueSim/CueSimLampSubsystem.h"
#include "Carla.h"
#include "Carla/BlueprintLibary/LightDefaultsJsonUtils.h"
#include "Carla/Game/CarlaEngine.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Carla/Game/CarlaStatics.h"
#include "Carla/Vehicle/CarlaWheeledVehicle.h"

#include <util/ue-header-guard-begin.h>
#include "Components/LightComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "HAL/IConsoleManager.h"
#include "Materials/MaterialInstance.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialParameterCollection.h"
#include "Materials/MaterialParameterCollectionInstance.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include <util/ue-header-guard-end.h>

#include <cmath>
#include <limits>

static TAutoConsoleVariable<int32> CVarCueSimLampController(
    TEXT("carla.CueSim.LampController"),
    0,
    TEXT("1 enables the CueSim lamp controller RPCs; vehicles given a schedule then ignore ")
    TEXT("the stock light state. 0 (default) keeps the stock behaviour and releases every ")
    TEXT("controlled vehicle."),
    ECVF_Default);

static TAutoConsoleVariable<float> CVarCueSimVisualTimePin(
    TEXT("carla.CueSim.LampController.VisualTimePin"),
    0.25f,
    TEXT("CarlaParameters.VisualTime while any vehicle is controlled, holding the material's ")
    TEXT("Sine(VisualTime) blinker gate open; negative leaves the stock clock running."),
    ECVF_Default);

namespace
{
  constexpr int32 MaxLogRows = 1 << 20;
  constexpr float LightDisplayScale = 10000.0f;  // VehicleLightIntensityDisplayScale
  const TCHAR *const ScheduleSchema = TEXT("cuesim-lamp-schedule-v1");
  const TCHAR *const CarlaParametersPath = TEXT("/Game/Carla/Blueprints/Game/CarlaParameters.CarlaParameters");

  // CueSim function key, M_VehicleLightsMaster scalar and Light Defaults group name.
  struct FFunctionName
  {
    const TCHAR *Key;
    const TCHAR *Group;
  };

  const FFunctionName FunctionNames[] = {
      {TEXT("position"), TEXT("Position")},
      {TEXT("low_beam"), TEXT("Low Beam")},
      {TEXT("high_beam"), TEXT("High Beam")},
      {TEXT("fog"), TEXT("Fog")},
      {TEXT("brake"), TEXT("Brake")},
      {TEXT("reverse"), TEXT("Reverse")},
      {TEXT("left_indicator"), TEXT("Left Blinker")},
      {TEXT("right_indicator"), TEXT("Right Blinker")},
  };

  constexpr int32 BrakeIndex = 4;
  constexpr int32 FirstIndicatorIndex = 6;
  const TCHAR *const CombinedKeys[] = {TEXT("brake_left"), TEXT("brake_right")};
  const TCHAR *const SlotSuffixes[] = {TEXT("_StopTurn_L"), TEXT("_StopTurn_R")};

  // Mirrors ComponentNameMatchesGroup in LightDefaultsJsonUtils.cpp, plus the
  // "Carla.Light.<Group>" tags ApplyVehicleLightDefaultsForCurrentState honours.
  bool LightBelongsTo(const ULightComponent &Light, const FString &Group)
  {
    if (Light.ComponentHasTag(FName(*(FString(TEXT("Carla.Light.")) + Group))))
    {
      return true;
    }
    const FString Lower = Light.GetName().ToLower();
    if (Group == TEXT("Left Blinker"))
      return Lower.Contains(TEXT("blinker")) && Lower.Contains(TEXT("-l-"));
    if (Group == TEXT("Right Blinker"))
      return Lower.Contains(TEXT("blinker")) && Lower.Contains(TEXT("-r-"));
    if (Group == TEXT("Low Beam"))
      return Lower.Contains(TEXT("low_beam"));
    if (Group == TEXT("High Beam"))
      return Lower.Contains(TEXT("high_beam"));
    if (Group == TEXT("Position"))
      return Lower.Contains(TEXT("position")) && !Lower.Contains(TEXT("plate"));
    if (Group == TEXT("Fog"))
      return Lower.Contains(TEXT("fog"));
    if (Group == TEXT("Brake"))
      return Lower.Contains(TEXT("brake"));
    if (Group == TEXT("Reverse"))
      return Lower.Contains(TEXT("reverse"));
    return false;
  }

  // Same base-material test as CollectVehicleEmissiveSlots in LightDefaultsJsonUtils.cpp.
  bool IsVehicleLightsMaterial(UMaterialInterface *Material)
  {
    if (UMaterialInstanceDynamic *Dynamic = Cast<UMaterialInstanceDynamic>(Material))
    {
      Material = Dynamic->Parent;
    }
    UMaterialInterface *Base = Material;
    for (int32 Depth = 0; Depth < 10; ++Depth)
    {
      UMaterialInstance *Instance = Cast<UMaterialInstance>(Base);
      if (Instance == nullptr || Instance->Parent == nullptr)
      {
        break;
      }
      Base = Instance->Parent;
    }
    return Base != nullptr && Base->GetName().Contains(TEXT("VehicleLightsMaster"));
  }

  bool CheckKeys(const FJsonObject &Object, std::initializer_list<const TCHAR *> Allowed,
      const FString &Where, FString &Error)
  {
    for (const auto &Pair : Object.Values)
    {
      bool bKnown = false;
      for (const TCHAR *Key : Allowed)
      {
        bKnown |= Pair.Key == Key;
      }
      if (!bKnown)
      {
        Error = FString::Printf(TEXT("unknown key %s in %s"), *Pair.Key, *Where);
        return false;
      }
    }
    return true;
  }

  bool ReadNumber(const FJsonObject &Object, const TCHAR *Key, const FString &Where,
      double &Out, FString &Error)
  {
    if (!Object.HasTypedField<EJson::Number>(Key))
    {
      Error = FString::Printf(TEXT("%s needs a number %s"), *Where, Key);
      return false;
    }
    Out = Object.GetNumberField(Key);
    if (!std::isfinite(Out))
    {
      Error = FString::Printf(TEXT("%s.%s is not finite"), *Where, Key);
      return false;
    }
    return true;
  }

  bool ParseInterval(const FJsonObject &Object, const FString &Where,
      CueSimLamp::FInterval &Out, FString &Error)
  {
    if (!CheckKeys(Object, {TEXT("start_s"), TEXT("end_s"), TEXT("flash")}, Where, Error) ||
        !ReadNumber(Object, TEXT("start_s"), Where, Out.Start, Error))
    {
      return false;
    }
    Out.End = CueSimLamp::Forever;
    if (!Object.HasTypedField<EJson::Null>(TEXT("end_s")) &&
        !ReadNumber(Object, TEXT("end_s"), Where, Out.End, Error))
    {
      return false;
    }
    if (!(Out.End > Out.Start))
    {
      Error = Where + TEXT(" must end after it starts");
      return false;
    }
    Out.bFlashing = !Object.HasTypedField<EJson::Null>(TEXT("flash"));
    if (!Out.bFlashing)
    {
      return true;
    }
    const TSharedPtr<FJsonObject> *Flash = nullptr;
    if (!Object.TryGetObjectField(TEXT("flash"), Flash))
    {
      Error = Where + TEXT(".flash must be an object or null");
      return false;
    }
    const FString FlashWhere = Where + TEXT(".flash");
    if (!CheckKeys(**Flash, {TEXT("period_s"), TEXT("duty"), TEXT("phase")}, FlashWhere, Error) ||
        !ReadNumber(**Flash, TEXT("period_s"), FlashWhere, Out.Flash.Period, Error) ||
        !ReadNumber(**Flash, TEXT("duty"), FlashWhere, Out.Flash.Duty, Error) ||
        !ReadNumber(**Flash, TEXT("phase"), FlashWhere, Out.Flash.Phase, Error))
    {
      return false;
    }
    if (!(Out.Flash.Period >= 0.05) || !(Out.Flash.Duty > 0.0 && Out.Flash.Duty < 1.0) ||
        !(Out.Flash.Phase >= 0.0 && Out.Flash.Phase < 1.0))
    {
      Error = FlashWhere + TEXT(" needs period_s >= 0.05, 0 < duty < 1 and 0 <= phase < 1");
      return false;
    }
    return true;
  }

  bool ParseFunction(const FJsonObject &Object, const FString &Where,
      CueSimLamp::FFunction &Out, FString &Error)
  {
    double T90On = 0.0;
    double T90Off = 0.0;
    if (!CheckKeys(Object, {TEXT("t90_on_s"), TEXT("t90_off_s"), TEXT("level"), TEXT("intervals")},
            Where, Error) ||
        !ReadNumber(Object, TEXT("t90_on_s"), Where, T90On, Error) ||
        !ReadNumber(Object, TEXT("t90_off_s"), Where, T90Off, Error) ||
        !ReadNumber(Object, TEXT("level"), Where, Out.Level, Error))
    {
      return false;
    }
    if (T90On < 0.0 || T90Off < 0.0 || Out.Level < 0.0)
    {
      Error = Where + TEXT(" needs nonnegative t90 and level");
      return false;
    }
    Out.TauOn = CueSimLamp::TauFromT90(T90On);
    Out.TauOff = CueSimLamp::TauFromT90(T90Off);
    const TArray<TSharedPtr<FJsonValue>> *Intervals = nullptr;
    if (!Object.TryGetArrayField(TEXT("intervals"), Intervals))
    {
      Error = Where + TEXT(".intervals must be an array");
      return false;
    }
    for (int32 Index = 0; Index < Intervals->Num(); ++Index)
    {
      const TSharedPtr<FJsonObject> *Item = nullptr;
      const FString ItemWhere = FString::Printf(TEXT("%s.intervals[%d]"), *Where, Index);
      CueSimLamp::FInterval Interval;
      if (!(*Intervals)[Index]->TryGetObject(Item) ||
          !ParseInterval(**Item, ItemWhere, Interval, Error))
      {
        if (Error.IsEmpty())
        {
          Error = ItemWhere + TEXT(" must be an object");
        }
        return false;
      }
      if (!Out.Intervals.empty() && Interval.Start < Out.Intervals.back().End)
      {
        Error = ItemWhere + TEXT(" overlaps or precedes the previous interval");
        return false;
      }
      Out.Intervals.push_back(Interval);
    }
    return true;
  }

  FString Number(double Value)
  {
    return std::isfinite(Value) ? FString::Printf(TEXT("%.17g"), Value) : FString(TEXT("null"));
  }
}

bool UCueSimLampSubsystem::IsEnabled()
{
  return CVarCueSimLampController.GetValueOnGameThread() == 1;
}

bool UCueSimLampSubsystem::DoesSupportWorldType(const EWorldType::Type WorldType) const
{
  return WorldType == EWorldType::Game || WorldType == EWorldType::PIE;
}

TStatId UCueSimLampSubsystem::GetStatId() const
{
  RETURN_QUICK_DECLARE_CYCLE_STAT(UCueSimLampSubsystem, STATGROUP_Tickables);
}

void UCueSimLampSubsystem::Deinitialize()
{
  ReleaseAll();
  Logs.Empty();
  Overflowed.Empty();
  Super::Deinitialize();
}

FString UCueSimLampSubsystem::SetSchedule(
    ACarlaWheeledVehicle &Vehicle, uint32 ActorId, const FString &Payload)
{
  if (!IsEnabled())
  {
    return TEXT("the lamp controller is disabled (carla.CueSim.LampController=0)");
  }
  if (Payload.IsEmpty())
  {
    if (FControlled *Entry = Controlled.Find(ActorId))
    {
      Release(*Entry);
      Controlled.Remove(ActorId);
    }
    return FString();
  }

  TSharedPtr<FJsonObject> Root;
  if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Payload), Root) || !Root.IsValid())
  {
    return TEXT("the schedule is not a JSON object");
  }
  FString Error;
  FControlled Entry;
  if (!CheckKeys(*Root, {TEXT("schema"), TEXT("schedule_id"), TEXT("exposure"),
          TEXT("material_intensity"), TEXT("functions"), TEXT("wiring")}, TEXT("schedule"), Error))
  {
    return Error;
  }
  FString Schema;
  if (!Root->TryGetStringField(TEXT("schema"), Schema) || Schema != ScheduleSchema)
  {
    return FString::Printf(TEXT("schema must be %s"), ScheduleSchema);
  }
  if (!Root->TryGetStringField(TEXT("schedule_id"), Entry.ScheduleId) || Entry.ScheduleId.Len() != 64)
  {
    return TEXT("schedule_id must be a SHA256 hex digest");
  }
  if (Root->HasField(TEXT("wiring")))
  {
    FString Wiring;
    if (!Root->TryGetStringField(TEXT("wiring"), Wiring) ||
        (Wiring != TEXT("separate") && Wiring != TEXT("combined_stop_turn")))
    {
      return TEXT("wiring must be separate or combined_stop_turn");
    }
    Entry.bCombined = Wiring == TEXT("combined_stop_turn");
  }
  const TSharedPtr<FJsonObject> *Exposure = nullptr;
  if (!Root->TryGetObjectField(TEXT("exposure"), Exposure) ||
      !CheckKeys(**Exposure, {TEXT("duration_s"), TEXT("centre_offset_s")}, TEXT("exposure"), Error) ||
      !ReadNumber(**Exposure, TEXT("duration_s"), TEXT("exposure"), Entry.Exposure.Duration, Error) ||
      !ReadNumber(**Exposure, TEXT("centre_offset_s"), TEXT("exposure"), Entry.Exposure.CentreOffset, Error))
  {
    return Error.IsEmpty() ? FString(TEXT("exposure must be an object")) : Error;
  }
  if (Entry.Exposure.Duration < 0.0)
  {
    return TEXT("exposure.duration_s must be nonnegative");
  }

  float DefaultIntensity = 0.0f;
  TMap<FString, float> GroupIntensity;
  const bool bHasDefault = ULightDefaultsJsonUtils::LoadVehicleLightDefault(
      Vehicle.GetClass()->GetName(), DefaultIntensity, GroupIntensity);
  if (Root->HasTypedField<EJson::Null>(TEXT("material_intensity")))
  {
    if (!bHasDefault || !(DefaultIntensity > 0.0f))
    {
      return FString::Printf(
          TEXT("no Light Defaults intensity for %s; pass material_intensity"),
          *Vehicle.GetClass()->GetName());
    }
    Entry.MaterialIntensity = DefaultIntensity;
  }
  else if (!ReadNumber(*Root, TEXT("material_intensity"), TEXT("schedule"), Entry.MaterialIntensity, Error) ||
      !(Entry.MaterialIntensity > 0.0))
  {
    return Error.IsEmpty() ? FString(TEXT("material_intensity must be positive or null")) : Error;
  }

  const TSharedPtr<FJsonObject> *FunctionsObject = nullptr;
  if (!Root->TryGetObjectField(TEXT("functions"), FunctionsObject))
  {
    return TEXT("functions must be an object");
  }
  for (const auto &Pair : (*FunctionsObject)->Values)
  {
    int32 Found = INDEX_NONE;
    for (int32 Index = 0; Index < FunctionCount; ++Index)
    {
      Found = Pair.Key == FunctionNames[Index].Key ? Index : Found;
    }
    const TSharedPtr<FJsonObject> *Item = nullptr;
    if (Found == INDEX_NONE || !Pair.Value->TryGetObject(Item))
    {
      return FString::Printf(TEXT("unsupported lamp function %s"), *Pair.Key);
    }
    if (!ParseFunction(**Item, FString(*Pair.Key), Entry.Functions[Found], Error))
    {
      return Error;
    }
  }

  // Resolve the vehicle's lamp materials and real lights before taking control, so
  // the base light intensities are the Blueprint's own.
  Entry.Vehicle = &Vehicle;
  TArray<UStaticMeshComponent *> Meshes;
  Vehicle.GetComponents<UStaticMeshComponent>(Meshes);
  struct FSlot
  {
    UStaticMeshComponent *Mesh;
    int32 Index;
    int32 Side;
  };
  TArray<FSlot> Slots;
  bool bFoundSide[CombinedCount] = {};
  for (UStaticMeshComponent *Mesh : Meshes)
  {
    const TArray<FName> Names = Mesh->GetMaterialSlotNames();
    for (int32 Slot = 0; Slot < Mesh->GetNumMaterials(); ++Slot)
    {
      if (!IsVehicleLightsMaterial(Mesh->GetMaterial(Slot)))
      {
        continue;
      }
      int32 Side = INDEX_NONE;
      if (Entry.bCombined && Names.IsValidIndex(Slot))
      {
        for (int32 Candidate = 0; Candidate < CombinedCount; ++Candidate)
        {
          if (Names[Slot].ToString().EndsWith(SlotSuffixes[Candidate]))
          {
            Side = Candidate;
            bFoundSide[Candidate] = true;
          }
        }
      }
      Slots.Add({Mesh, Slot, Side});
    }
  }
  if (Slots.IsEmpty())
  {
    return TEXT("the vehicle has no M_VehicleLightsMaster material slot");
  }
  if (Entry.bCombined)
  {
    for (int32 Side = 0; Side < CombinedCount; ++Side)
    {
      if (!bFoundSide[Side])
      {
        return FString::Printf(TEXT("combined_stop_turn requires a lamp material slot ending in %s"),
            SlotSuffixes[Side]);
      }
      Entry.CombinedFunctions[Side] = CueSimLamp::Combined(
          Entry.Functions[BrakeIndex], Entry.Functions[FirstIndicatorIndex + Side]);
    }
  }
  for (const FSlot &Slot : Slots)
  {
    auto &Materials = Slot.Side == INDEX_NONE ? Entry.Materials : Entry.CombinedMaterials[Slot.Side];
    Materials.Add(Slot.Mesh->CreateAndSetMaterialInstanceDynamic(Slot.Index));
  }
  TArray<ULightComponent *> Lights;
  Vehicle.GetComponents<ULightComponent>(Lights);
  for (int32 Index = 0; Index < FunctionCount; ++Index)
  {
    const FString Group = FunctionNames[Index].Group;
    const float *Configured = GroupIntensity.Find(Group);
    for (ULightComponent *Light : Lights)
    {
      if (Light != nullptr && LightBelongsTo(*Light, Group))
      {
        Entry.Lights[Index].Add(Light);
        Entry.LightBaseIntensity[Index].Add(
            Configured != nullptr ? *Configured * LightDisplayScale : Light->Intensity);
      }
    }
  }

  if (FControlled *Previous = Controlled.Find(ActorId))
  {
    // A new schedule replaces the old one from the next tick; the vehicle stays controlled.
    for (int32 Index = 0; Index < FunctionCount; ++Index)
    {
      Entry.LightBaseIntensity[Index] = Previous->LightBaseIntensity[Index];
      Entry.Lights[Index] = Previous->Lights[Index];
    }
  }
  else
  {
    Vehicle.SetCueSimLampControlled(true);
  }
  Controlled.Add(ActorId, MoveTemp(Entry));
  return FString();
}

bool UCueSimLampSubsystem::TakeLog(uint32 ActorId, FString &OutJsonLines, FString &OutError)
{
  TArray<FString> Rows;
  Logs.RemoveAndCopyValue(ActorId, Rows);
  if (Overflowed.Remove(ActorId) > 0)
  {
    OutError = FString::Printf(TEXT("the lamp log overflowed %d rows; rows dropped"), MaxLogRows);
    return false;
  }
  OutJsonLines = FString::Join(Rows, TEXT(""));
  return true;
}

void UCueSimLampSubsystem::Release(FControlled &Entry)
{
  for (int32 Index = 0; Index < FunctionCount; ++Index)
  {
    for (int32 Light = 0; Light < Entry.Lights[Index].Num(); ++Light)
    {
      if (ULightComponent *Component = Entry.Lights[Index][Light].Get())
      {
        Component->SetIntensity(Entry.LightBaseIntensity[Index][Light]);
      }
    }
  }
  if (ACarlaWheeledVehicle *Vehicle = Entry.Vehicle.Get())
  {
    Vehicle->SetCueSimLampControlled(false);
  }
}

void UCueSimLampSubsystem::ReleaseAll()
{
  for (auto &Pair : Controlled)
  {
    Release(Pair.Value);
  }
  Controlled.Empty();
}

float UCueSimLampSubsystem::PinVisualTime()
{
  const float Pin = CVarCueSimVisualTimePin.GetValueOnGameThread();
  if (CarlaParameters == nullptr)
  {
    CarlaParameters = LoadObject<UMaterialParameterCollection>(nullptr, CarlaParametersPath);
  }
  UMaterialParameterCollectionInstance *Instance =
      CarlaParameters != nullptr ? GetWorld()->GetParameterCollectionInstance(CarlaParameters) : nullptr;
  if (Instance == nullptr)
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (Pin >= 0.0f)
  {
    // UCarlaEpisode::TickTimers set the running clock at OnWorldTickStart; this
    // later write is what the frame renders. The episode's own VisualGameTime,
    // snapshots and recorder are unchanged.
    Instance->SetScalarParameterValue(FName(TEXT("VisualTime")), Pin);
  }
  float Value = std::numeric_limits<float>::quiet_NaN();
  Instance->GetScalarParameterValue(FName(TEXT("VisualTime")), Value);
  return Value;
}

void UCueSimLampSubsystem::Tick(float DeltaTime)
{
  Super::Tick(DeltaTime);
  if (!IsEnabled())
  {
    ReleaseAll();
    return;
  }
  if (Controlled.IsEmpty())
  {
    return;
  }
  UCarlaEpisode *Episode = UCarlaStatics::GetCurrentEpisode(GetWorld());
  if (Episode == nullptr || Episode->GetWorld() != GetWorld())
  {
    return;
  }
  const double SimTime = Episode->GetElapsedGameTime();
  const uint64 Frame = FCarlaEngine::GetFrameCounter();
  const float VisualTime = PinVisualTime();
  for (auto It = Controlled.CreateIterator(); It; ++It)
  {
    if (!It.Value().Vehicle.IsValid())
    {
      It.RemoveCurrent();
      continue;
    }
    Apply(It.Key(), It.Value(), Frame, SimTime, VisualTime);
  }
}

void UCueSimLampSubsystem::Apply(
    uint32 ActorId, FControlled &Entry, uint64 Frame, double SimTime, float VisualTime)
{
  TArray<FString> &Rows = Logs.FindOrAdd(ActorId);
  const int32 RowCount = FunctionCount + (Entry.bCombined ? CombinedCount : 0);
  for (int32 Index = 0; Index < RowCount; ++Index)
  {
    const bool bCombined = Index >= FunctionCount;
    const int32 Side = Index - FunctionCount;
    const CueSimLamp::FFunction &Function = bCombined ? Entry.CombinedFunctions[Side] : Entry.Functions[Index];
    const CueSimLamp::FSample Sample = CueSimLamp::Evaluate(Function, Entry.Exposure, SimTime);
    const float Written =
        static_cast<float>(Entry.MaterialIntensity * Function.Level * Sample.ExposureFraction);

    // Vehicles mix lamp masters (sirens, per-lamp materials): a parameter counts as found when
    // any lamp slot has it, and only those slots are written, read back and compared.
    int32 Slots = 0;
    bool bFound = false;
    bool bAgree = true;
    float Readback = std::numeric_limits<float>::quiet_NaN();
    const auto Write = [&](const auto &Materials, int32 ParameterIndex)
    {
      const FName Parameter(FunctionNames[ParameterIndex].Group);
      const FHashedMaterialParameterInfo Info(Parameter);
      for (const auto &Weak : Materials)
      {
        UMaterialInstanceDynamic *Material = Weak.Get();
        if (Material == nullptr)
        {
          bAgree = false;
          continue;
        }
        float Default = 0.0f;
        if (Material->Parent == nullptr ||
            !Material->Parent->GetScalarParameterDefaultValue(Info, Default))
        {
          continue;
        }
        bFound = true;
        Material->SetScalarParameterValue(Parameter, Written);
        float Value = std::numeric_limits<float>::quiet_NaN();
        const bool bRead = Material->GetScalarParameterValue(Info, Value, /*bOveriddenOnly=*/true);
        bAgree &= bRead && Value == Written;
        Readback = Slots == 0 ? Value : Readback;
        ++Slots;
      }
    };
    if (bCombined)
    {
      Write(Entry.CombinedMaterials[Side], BrakeIndex);
      Write(Entry.CombinedMaterials[Side], FirstIndicatorIndex + Side);
    }
    else
    {
      Write(Entry.Materials, Index);
      for (int32 OtherSide = 0; OtherSide < CombinedCount; ++OtherSide)
      {
        if (Index != BrakeIndex && Index != FirstIndicatorIndex + OtherSide)
        {
          Write(Entry.CombinedMaterials[OtherSide], Index);
        }
      }
    }

    float LightIntensity = 0.0f;
    const int32 LightCount = bCombined ? 0 : Entry.Lights[Index].Num();
    for (int32 Light = 0; Light < LightCount; ++Light)
    {
      ULightComponent *Component = Entry.Lights[Index][Light].Get();
      if (Component == nullptr)
      {
        continue;
      }
      const float Intensity = static_cast<float>(
          Entry.LightBaseIntensity[Index][Light] * Function.Level * Sample.ExposureFraction);
      Component->SetIntensity(Intensity);
      Component->SetVisibility(Sample.ExposureFraction > 0.0);
      LightIntensity += Component->Intensity;
    }

    if (Rows.Num() >= MaxLogRows)
    {
      Overflowed.Add(ActorId);
      continue;
    }
    Rows.Add(FString::Printf(
        TEXT("{\"frame\":%llu,\"sim_time\":%s,\"actor_id\":%u,\"schedule_id\":\"%s\",")
        TEXT("\"function\":\"%s\",\"electrical\":%s,\"level\":%s,\"exposure_fraction\":%s,")
        TEXT("\"material_intensity\":%s,\"written\":%s,\"readback\":%s,\"parameter_found\":%s,")
        TEXT("\"slots\":%d,\"slots_agree\":%s,\"lights\":%d,\"light_intensity\":%s,")
        TEXT("\"visual_time\":%s}\n"),
        static_cast<unsigned long long>(Frame), *Number(SimTime), ActorId, *Entry.ScheduleId,
        bCombined ? CombinedKeys[Side] : FunctionNames[Index].Key, Sample.bElectrical ? TEXT("true") : TEXT("false"),
        *Number(Sample.LevelAtCentre), *Number(Sample.ExposureFraction),
        *Number(Entry.MaterialIntensity), *Number(Written), *Number(Readback),
        bFound && Slots > 0 ? TEXT("true") : TEXT("false"), Slots,
        bAgree ? TEXT("true") : TEXT("false"), LightCount,
        *Number(LightIntensity), *Number(VisualTime)));
  }
}
