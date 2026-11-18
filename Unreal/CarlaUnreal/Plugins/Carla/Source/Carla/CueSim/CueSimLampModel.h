// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

// CueSim lamp timeline: the C++ twin of CueSim's reference formula
// (`cuesim.lamps.timeline`). Plain C++17 without Unreal types so that the same
// header compiles in the plugin and in CueSim's standalone parity driver. Keep
// every arithmetic expression in the same order as the Python reference: the
// edge times must be bit-identical, and the ramps differ only by libm `exp`.
//
// electrical(t)  square wave or steady drive per schedule interval, [rise, fall)
// L(t)           first-order response, dL/dt = (electrical - L) / tau, L = 0 before the
//                first edge; tau = t90 / ln(10) (on and off separately), tau = 0 is an LED
// E_k            (1/T) * integral of L over [c - T/2, c + T/2], c = t_k + centre offset;
//                L(c) when T = 0

#pragma once

#include <cmath>
#include <limits>
#include <vector>

// Each operation rounds on its own, as in Python: no fused multiply-add contraction.
#if defined(__clang__)
#define CUESIM_LAMP_EXACT _Pragma("clang fp contract(off)")
#else
#define CUESIM_LAMP_EXACT
#endif

namespace CueSimLamp
{
  constexpr double Ln10 = 2.302585092994046;
  constexpr double Forever = std::numeric_limits<double>::infinity();

  struct FFlash
  {
    double Period = 0.0;  // seconds, > 0
    double Duty = 0.5;    // lit fraction of a cycle, (0, 1)
    double Phase = 0.0;   // cycle fraction at the interval start, [0, 1)
  };

  struct FInterval
  {
    double Start = 0.0;
    double End = Forever;
    bool bFlashing = false;
    FFlash Flash;
  };

  struct FFunction
  {
    double TauOn = 0.0;
    double TauOff = 0.0;
    double Level = 1.0;
    std::vector<FInterval> Intervals;  // sorted, non-overlapping
  };

  struct FExposure
  {
    double Duration = 0.0;
    double CentreOffset = 0.0;
  };

  struct FEdge
  {
    double Time;
    bool bRise;
  };

  struct FSample
  {
    bool bElectrical = false;
    double LevelAtCentre = 0.0;
    double ExposureFraction = 0.0;
  };

  inline double TauFromT90(double T90)
  {
    return T90 / Ln10;
  }

  /// Rise and fall edges with time <= Until, in schedule order.
  inline std::vector<FEdge> Edges(const FFunction &Function, double Until)
  {
    CUESIM_LAMP_EXACT
    std::vector<FEdge> Result;
    for (const FInterval &Interval : Function.Intervals)
    {
      if (Interval.Start > Until)
      {
        break;
      }
      if (!Interval.bFlashing)
      {
        Result.push_back({Interval.Start, true});
        if (Interval.End != Forever)
        {
          Result.push_back({Interval.End, false});
        }
        continue;
      }
      const FFlash &Flash = Interval.Flash;
      for (long long N = 0;; ++N)
      {
        const double Offset = (static_cast<double>(N) - Flash.Phase) * Flash.Period;
        const double Cycle = Interval.Start + Offset;
        const double Rise = Cycle > Interval.Start ? Cycle : Interval.Start;
        if (Rise >= Interval.End || Rise > Until)
        {
          break;
        }
        const double LitLength = Flash.Duty * Flash.Period;
        const double LitEnd = Cycle + LitLength;
        const double Fall = LitEnd < Interval.End ? LitEnd : Interval.End;
        if (Rise < Fall)
        {
          Result.push_back({Rise, true});
          Result.push_back({Fall, false});
        }
      }
    }
    return Result;
  }

  namespace Detail
  {
    inline double Value(double Target, double Initial, double Tau, double Elapsed)
    {
      CUESIM_LAMP_EXACT
      if (Tau == 0.0)
      {
        return Target;
      }
      return Target + (Initial - Target) * std::exp(-Elapsed / Tau);
    }

    // Integral over [A, B] of a segment that starts at T0 with level Initial.
    inline double Integral(double Target, double Initial, double Tau, double T0, double A, double B)
    {
      CUESIM_LAMP_EXACT
      if (Tau == 0.0)
      {
        return Target * (B - A);
      }
      return Target * (B - A) +
          (Initial - Target) * Tau * (std::exp(-(A - T0) / Tau) - std::exp(-(B - T0) / Tau));
    }
  }

  /// Electrical state, L at the window centre, and the exposure-integrated fraction E_k.
  inline FSample Evaluate(const FFunction &Function, const FExposure &Exposure, double FrameTime)
  {
    CUESIM_LAMP_EXACT
    const double Centre = FrameTime + Exposure.CentreOffset;
    const double Half = Exposure.Duration / 2.0;
    const double A = Centre - Half;
    const double B = Centre + Half;
    const std::vector<FEdge> All = Edges(Function, B);

    FSample Sample;
    double SegmentStart = -Forever;
    double Initial = 0.0;
    bool bOn = false;
    double Integral = 0.0;
    bool bCentreDone = false;
    // Walk segments [SegmentStart, next edge) and close them at the next edge.
    for (std::size_t Index = 0; Index <= All.size(); ++Index)
    {
      const double SegmentEnd = Index < All.size() ? All[Index].Time : Forever;
      const double Tau = bOn ? Function.TauOn : Function.TauOff;
      const double Target = bOn ? 1.0 : 0.0;
      if (!bCentreDone && Centre < SegmentEnd)
      {
        Sample.bElectrical = bOn;
        Sample.LevelAtCentre = SegmentStart == -Forever
            ? Target
            : Detail::Value(Target, Initial, Tau, Centre - SegmentStart);
        bCentreDone = true;
      }
      if (Exposure.Duration > 0.0 && SegmentStart < B && SegmentEnd > A)
      {
        const double Lo = SegmentStart > A ? SegmentStart : A;
        const double Hi = SegmentEnd < B ? SegmentEnd : B;
        if (Hi > Lo)
        {
          Integral += SegmentStart == -Forever
              ? Target * (Hi - Lo)
              : Detail::Integral(Target, Initial, Tau, SegmentStart, Lo, Hi);
        }
      }
      if (Index == All.size())
      {
        break;
      }
      Initial = SegmentStart == -Forever
          ? Target
          : Detail::Value(Target, Initial, Tau, SegmentEnd - SegmentStart);
      SegmentStart = SegmentEnd;
      bOn = All[Index].bRise;
    }
    Sample.ExposureFraction =
        Exposure.Duration > 0.0 ? Integral / Exposure.Duration : Sample.LevelAtCentre;
    return Sample;
  }
}

#undef CUESIM_LAMP_EXACT
