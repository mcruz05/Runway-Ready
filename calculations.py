"""Physics model for aircraft take-off runway feasibility.
"""

from dataclasses import dataclass
from math import cos, radians, sin, sqrt

G = 9.80665
SPECIFIC_GAS_CONSTANT_AIR = 287.1


@dataclass(frozen=True)
class Runway:
    """Runway geometry and wind alignment."""

    length_m: float
    heading_deg: float
    slope_percent: float = 0.0


@dataclass(frozen=True)
class Atmosphere:
    """Local atmospheric and wind conditions.
        Wind direction follows convention.
    """

    temperature_c: float
    pressure_pa: float
    density: float | None = None
    wind_from_deg: float = 0.0
    wind_speed_mps: float = 0.0

    @property
    def effective_density(self) -> float:

        if self.density is not None:
            return self.density
        
        temperature_k = self.temperature_c + 273.15
        return self.pressure_pa / (SPECIFIC_GAS_CONSTANT_AIR * temperature_k)


@dataclass(frozen=True)
class Aircraft:
    """Aircraft parameters."""

    mass: float
    wing_area: float
    CL_max: float
    thrust: float
    CD: float
    rolling_friction_coefficient: float


@dataclass(frozen=True)
class TakeoffResult:
    """Main output of the go/no-go model."""

    estimated_takeoff_distance_m: float
    remaining_runway_margin_m: float
    go_no_go: str
    reason: str
    stall_speed_mps: float
    liftoff_true_airspeed: float
    liftoff_ground_speed: float
    headwind_component: float
    crosswind_component: float
    density: float


def wind_components(runway: Runway, atmosphere: Atmosphere) -> tuple[float, float]:
    """Return headwind and crosswind components in m/s.
    Positive headwind helps take-off.
    """

    angle_difference_rad = radians(atmosphere.wind_from_deg - runway.heading_deg)
    headwind_mps = atmosphere.wind_speed_mps * cos(angle_difference_rad)
    crosswind_mps = atmosphere.wind_speed_mps * sin(angle_difference_rad)
    return headwind_mps, crosswind_mps


def calculate_stall_speed(aircraft: Aircraft, density: float) -> float:
    """Calculate stall speed from lift equals weight."""

    weight_n = aircraft.mass * G
    return sqrt((2.0 * weight_n)/(density*aircraft.wing_area*aircraft.CL_max))


def estimate_takeoff_distance(runway: Runway,atmosphere: Atmosphere,aircraft: Aircraft,
    liftoff_speed_factor: float = 1.2,lift_coefficient_during_roll_factor: float = 0.7,
    speed_step_mps: float = 0.25,) -> TakeoffResult:
    """Estimate take-off distance and return a go/no-go result.

    Integrates acceleration over ground speed. At each speed step:
    net force = thrust - aerodynamic drag - rolling friction - slope force
    Lift reduces the normal force on the wheels, reducing rolling friction.
    """

    density = atmosphere.effective_density
    headwind_mps, crosswind_mps = wind_components(runway, atmosphere)
    stall_speed_mps = calculate_stall_speed(aircraft, density)
    liftoff_true_airspeed = liftoff_speed_factor * stall_speed_mps
    liftoff_ground_speed = max(liftoff_true_airspeed - headwind_mps, 0.0)

    if liftoff_ground_speed == 0.0:
        return TakeoffResult(
            estimated_takeoff_distance_m=0.0,
            remaining_runway_margin_m=runway.length_m,
            go_no_go="GO",
            reason="Headwind already provides the required lift-off airspeed.",
            stall_speed_mps=stall_speed_mps,
            liftoff_true_airspeed=liftoff_true_airspeed,
            liftoff_ground_speed=liftoff_ground_speed,
            headwind_component=headwind_mps,
            crosswind_component=crosswind_mps,
            density=density,
        )

    weight_n = aircraft.mass * G
    lift_coefficient_roll = (
        aircraft.CL_max * lift_coefficient_during_roll_factor
    )
    slope_force_n = weight_n * (runway.slope_percent / 100.0)

    distance_m = 0.0
    ground_speed_mps = 0.0

    while ground_speed_mps < liftoff_ground_speed:
        next_ground_speed_mps = min(
            ground_speed_mps + speed_step_mps,
            liftoff_ground_speed,
        )
        middle_ground_speed_mps = 0.5 * (ground_speed_mps + next_ground_speed_mps)
        middle_airspeed_mps = max(middle_ground_speed_mps + headwind_mps, 0.0)

        dynamic_pressure_pa = 0.5 * density * middle_airspeed_mps**2
        lift_n = dynamic_pressure_pa * aircraft.wing_area * lift_coefficient_roll
        drag_n = (
            dynamic_pressure_pa * aircraft.wing_area * aircraft.CD
        )
        normal_force_n = max(weight_n - lift_n, 0.0)
        rolling_friction_n = aircraft.rolling_friction_coefficient * normal_force_n
        net_force_n = (
            aircraft.thrust - drag_n - rolling_friction_n - slope_force_n
        )
        acceleration_mps2 = net_force_n / aircraft.mass

        if acceleration_mps2 <= 0.0:
            return TakeoffResult(
                estimated_takeoff_distance_m=float("inf"),
                remaining_runway_margin_m=float("-inf"),
                go_no_go="NO-GO",
                reason=(
                    "Aircraft cannot continue accelerating to lift-off speed "
                    "with these inputs."
                ),
                stall_speed_mps=stall_speed_mps,
                liftoff_true_airspeed=liftoff_true_airspeed,
                liftoff_ground_speed=liftoff_ground_speed,
                headwind_component=headwind_mps,
                crosswind_component=crosswind_mps,
                density=density,
            )

        distance_m += (
            next_ground_speed_mps**2 - ground_speed_mps**2
        ) / (2.0 * acceleration_mps2)
        ground_speed_mps = next_ground_speed_mps

    remaining_runway_margin_m = runway.length_m - distance_m
    can_take_off = remaining_runway_margin_m >= 0.0

    return TakeoffResult(
        estimated_takeoff_distance_m=distance_m,
        remaining_runway_margin_m=remaining_runway_margin_m,
        go_no_go="GO" if can_take_off else "NO-GO",
        reason=(
            "Estimated take-off distance is within available runway length."
            if can_take_off
            else "Estimated take-off distance exceeds available runway length."
        ),
        stall_speed_mps=stall_speed_mps,
        liftoff_true_airspeed=liftoff_true_airspeed,
        liftoff_ground_speed=liftoff_ground_speed,
        headwind_component=headwind_mps,
        crosswind_component=crosswind_mps,
        density=density,
    )


def demo_takeoff_check() -> TakeoffResult:
    """Run a hard-coded example for the first model version."""

    runway = Runway(
        length_m=3705.0,
        heading_deg=21.5,
        slope_percent=0.0 )
    
    atmosphere = Atmosphere(
        temperature_c=25.0,
        pressure_pa=101325.0,
        density=1.225,
        wind_from_deg=330.0,
        wind_speed_mps=3.6)
    
    aircraft = Aircraft(
        mass=242000.0,
        wing_area=372.0,
        CL_max=1.8,
        thrust=640000.,
        CD=0.03,
        rolling_friction_coefficient=0.015)

    return estimate_takeoff_distance(runway, atmosphere, aircraft)


if __name__ == "__main__":
    result = demo_takeoff_check()
    print(f"Estimated take-off distance: {result.estimated_takeoff_distance_m:.1f} m")
    print(f"Remaining runway margin: {result.remaining_runway_margin_m:.1f} m")
    print(f"Go/no-go result: {result.go_no_go}")
    print(f"Reason: {result.reason}")
