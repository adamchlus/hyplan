"""Tests for hyplan.flight_line."""

import datetime

import numpy as np
import pytest
from shapely.geometry import box

from hyplan.exceptions import HyPlanTypeError, HyPlanValueError
from hyplan.flight_line import FlightLine, to_gdf
from hyplan.units import ureg
from hyplan.waypoint import Waypoint


class TestFlightLineCreation:
    def test_start_length_azimuth(self, sample_flight_line):
        fl = sample_flight_line
        assert fl.lat1 == pytest.approx(34.05)
        assert fl.lon1 == pytest.approx(-118.25)
        assert fl.length.m_as("meter") == pytest.approx(50000, rel=0.01)
        assert fl.site_name == "Test Line"

    def test_center_length_azimuth(self):
        fl = FlightLine.center_length_azimuth(
            lat=34.0,
            lon=-118.0,
            length=ureg.Quantity(20000, "meter"),
            az=0.0,
            altitude_msl=ureg.Quantity(5000, "meter"),
        )
        # Center should be near (34, -118)
        mid_lat = (fl.lat1 + fl.lat2) / 2
        assert mid_lat == pytest.approx(34.0, abs=0.01)
        assert fl.length.m_as("meter") == pytest.approx(20000, rel=0.01)

    def test_altitude_stored(self, sample_flight_line):
        assert sample_flight_line.altitude_msl.m_as("meter") == pytest.approx(6000)


class TestFlightLineProperties:
    def test_azimuth(self, sample_flight_line):
        assert sample_flight_line.az12.magnitude == pytest.approx(45.0, abs=1.0)

    def test_reverse_azimuth(self, sample_flight_line):
        assert sample_flight_line.az21.magnitude == pytest.approx(225.0, abs=1.0)

    def test_waypoints(self, sample_flight_line):
        wp1 = sample_flight_line.waypoint1
        wp2 = sample_flight_line.waypoint2
        assert wp1.latitude == pytest.approx(34.05)
        assert wp2.latitude != wp1.latitude  # should have moved


class TestFlightLineOperations:
    def test_reverse(self, sample_flight_line):
        rev = sample_flight_line.reverse()
        assert rev.lat1 == pytest.approx(sample_flight_line.lat2)
        assert rev.lon1 == pytest.approx(sample_flight_line.lon2)
        assert rev.lat2 == pytest.approx(sample_flight_line.lat1)

    def test_split_by_length(self, sample_flight_line):
        # 50 km line split into 10 km segments → 5 parts
        parts = sample_flight_line.split_by_length(ureg.Quantity(10000, "meter"))
        assert isinstance(parts, list)
        assert len(parts) == 5
        for part in parts:
            assert part.length.m_as("meter") == pytest.approx(10000, rel=0.01)

    def test_offset_across(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(1000, "meter"))
        # Should be a parallel line roughly 1 km away
        assert offset_fl.length.m_as("meter") == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_rotate_around_midpoint(self, sample_flight_line):
        rotated = sample_flight_line.rotate_around_midpoint(90.0)
        # Rotation should change azimuth; just verify it changed
        assert rotated.az12.magnitude != pytest.approx(45.0, abs=5.0)

    def test_to_dict(self, sample_flight_line):
        d = sample_flight_line.to_dict()
        assert "lat1" in d
        assert "lon1" in d
        assert "altitude_msl" in d

    def test_to_geojson(self, sample_flight_line):
        gj = sample_flight_line.to_geojson()
        assert gj["type"] == "Feature"
        assert gj["geometry"]["type"] == "LineString"

    def test_from_geojson_roundtrip(self, sample_flight_line):
        gj = sample_flight_line.to_geojson()
        restored = FlightLine.from_geojson(gj)
        assert restored.lat1 == pytest.approx(sample_flight_line.lat1, abs=1e-4)
        assert restored.lon1 == pytest.approx(sample_flight_line.lon1, abs=1e-4)
        assert restored.lat2 == pytest.approx(sample_flight_line.lat2, abs=1e-4)
        assert restored.lon2 == pytest.approx(sample_flight_line.lon2, abs=1e-4)
        assert restored.altitude_msl.m_as("meter") == pytest.approx(
            sample_flight_line.altitude_msl.m_as("meter"), rel=1e-3
        )
        assert restored.site_name == sample_flight_line.site_name

    def test_from_geojson_preserves_metadata(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(10, "km"), az=90,
            altitude_msl=ureg.Quantity(3000, "meter"),
            site_name="Test Line",
            site_description="A test",
            investigator="Dr. Smith",
        )
        gj = fl.to_geojson()
        restored = FlightLine.from_geojson(gj)
        assert restored.site_name == "Test Line"
        assert restored.site_description == "A test"
        assert restored.investigator == "Dr. Smith"


class TestSplitByLength:
    def test_line_shorter_than_max_returns_self(self, short_flight_line):
        """A line shorter than max_length should return itself unchanged."""
        parts = short_flight_line.split_by_length(ureg.Quantity(20000, "meter"))
        assert len(parts) == 1
        assert parts[0].length.m_as("meter") == pytest.approx(
            short_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_segments_cover_full_length(self, sample_flight_line):
        """Sum of segment lengths should equal total line length."""
        parts = sample_flight_line.split_by_length(ureg.Quantity(12000, "meter"))
        total = sum(p.length.m_as("meter") for p in parts)
        assert total == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_segments_preserve_altitude(self, sample_flight_line):
        parts = sample_flight_line.split_by_length(ureg.Quantity(10000, "meter"))
        for part in parts:
            assert part.altitude_msl.m_as("meter") == pytest.approx(6000)

    def test_split_with_gap(self, sample_flight_line):
        """Splitting with a gap should produce fewer segments than without."""
        parts_no_gap = sample_flight_line.split_by_length(ureg.Quantity(10000, "meter"))
        parts_with_gap = sample_flight_line.split_by_length(
            ureg.Quantity(10000, "meter"),
            gap_length=ureg.Quantity(5000, "meter"),
        )
        # With gaps the segments don't cover the full line, so fewer segments
        assert len(parts_with_gap) <= len(parts_no_gap)
        # Each segment should still be at most max_length
        for part in parts_with_gap:
            assert part.length.m_as("meter") <= 10001  # tolerance

    def test_split_invalid_max_length(self, sample_flight_line):
        """Zero or negative max_length should raise an error."""
        with pytest.raises(Exception):
            sample_flight_line.split_by_length(ureg.Quantity(0, "meter"))
        with pytest.raises(Exception):
            sample_flight_line.split_by_length(ureg.Quantity(-100, "meter"))

    def test_split_negative_gap(self, sample_flight_line):
        """Negative gap_length should raise an error."""
        with pytest.raises(Exception):
            sample_flight_line.split_by_length(
                ureg.Quantity(10000, "meter"),
                gap_length=ureg.Quantity(-500, "meter"),
            )

    def test_segment_names_include_index(self, sample_flight_line):
        """Each segment should get a unique name with its index."""
        parts = sample_flight_line.split_by_length(ureg.Quantity(10000, "meter"))
        for i, part in enumerate(parts):
            assert f"seg_{i}" in part.site_name

    def test_segments_follow_geodesic_on_long_high_latitude_line(self):
        """Splitting a 1000 km east-west line at 60°N must keep segments on
        the parent geodesic: the last segment's endpoint lands on the
        parent line's endpoint instead of drifting off along the initial
        azimuth."""
        import pymap3d.vincenty

        fl = FlightLine.start_length_azimuth(
            lat1=60.0, lon1=0.0,
            length=ureg.Quantity(1000, "km"), az=90.0,
            altitude_msl=ureg.Quantity(10000, "meter"),
            site_name="Geodesic",
        )
        parts = fl.split_by_length(ureg.Quantity(100, "km"))
        assert len(parts) == 10

        last = parts[-1]
        miss_m, _ = pymap3d.vincenty.vdist(last.lat2, last.lon2, fl.lat2, fl.lon2)
        assert float(miss_m) < 10.0

        # Consecutive segments stay contiguous
        import itertools
        for prev, nxt in itertools.pairwise(parts):
            gap_m, _ = pymap3d.vincenty.vdist(prev.lat2, prev.lon2, nxt.lat1, nxt.lon1)
            assert float(gap_m) < 1.0


class TestOffsetAcross:
    def test_positive_offset_preserves_length(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(2000, "meter"))
        assert offset_fl.length.m_as("meter") == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_negative_offset_preserves_length(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(-2000, "meter"))
        assert offset_fl.length.m_as("meter") == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_offset_preserves_altitude(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(500, "meter"))
        assert offset_fl.altitude_msl.m_as("meter") == pytest.approx(6000)

    def test_zero_offset_returns_same_position(self, sample_flight_line):
        offset_fl = sample_flight_line.offset_across(ureg.Quantity(0, "meter"))
        assert offset_fl.lat1 == pytest.approx(sample_flight_line.lat1, abs=1e-4)
        assert offset_fl.lon1 == pytest.approx(sample_flight_line.lon1, abs=1e-4)

    def test_plain_float_assumed_meters(self, sample_flight_line):
        """A plain float should be treated as meters."""
        offset_fl = sample_flight_line.offset_across(1000.0)
        assert offset_fl.length.m_as("meter") == pytest.approx(
            sample_flight_line.length.m_as("meter"), rel=0.01
        )

    def test_opposite_offsets_are_symmetric(self, sample_flight_line):
        """Positive and negative offsets should move the line to opposite sides."""
        right = sample_flight_line.offset_across(ureg.Quantity(1000, "meter"))
        left = sample_flight_line.offset_across(ureg.Quantity(-1000, "meter"))
        # The midpoints should be on opposite sides of the original
        mid_orig_lat = (sample_flight_line.lat1 + sample_flight_line.lat2) / 2
        mid_right_lat = (right.lat1 + right.lat2) / 2
        mid_left_lat = (left.lat1 + left.lat2) / 2
        # One should be above and one below the original (for az=45)
        assert (mid_right_lat - mid_orig_lat) != pytest.approx(
            mid_left_lat - mid_orig_lat, abs=1e-5
        )


class TestRotateAroundMidpoint:
    def test_rotation_changes_azimuth(self, sample_flight_line):
        rotated = sample_flight_line.rotate_around_midpoint(45.0)
        # Az should have changed significantly from original ~45 degrees
        assert rotated.az12.magnitude != pytest.approx(
            sample_flight_line.az12.magnitude, abs=10.0
        )

    def test_rotation_preserves_length(self, short_flight_line):
        """Rotation reconstructs the line geodesically, so length is
        preserved regardless of latitude or rotation angle."""
        rotated = short_flight_line.rotate_around_midpoint(10.0)
        assert rotated.length.m_as("meter") == pytest.approx(
            short_flight_line.length.m_as("meter"), rel=1e-3
        )

    def test_high_latitude_rotation_geodesic(self):
        """At 60N a 90 deg CCW rotation must preserve length and decrease
        the compass azimuth by 90 deg (compass azimuth increases CW)."""
        fl = FlightLine.center_length_azimuth(
            lat=60.0, lon=10.0,
            length=ureg.Quantity(50, "km"), az=0.0,
            altitude_msl=ureg.Quantity(3000, "meter"),
        )
        rotated = fl.rotate_around_midpoint(90.0)
        assert rotated.length.m_as("meter") == pytest.approx(50000, rel=1e-3)
        assert rotated.az12.magnitude % 360 == pytest.approx(270.0, abs=0.5)

    def test_rotation_preserves_altitude(self, sample_flight_line):
        rotated = sample_flight_line.rotate_around_midpoint(90.0)
        assert rotated.altitude_msl.m_as("meter") == pytest.approx(6000)

    def test_360_rotation_returns_original(self, sample_flight_line):
        rotated = sample_flight_line.rotate_around_midpoint(360.0)
        assert rotated.lat1 == pytest.approx(sample_flight_line.lat1, abs=1e-4)
        assert rotated.lon1 == pytest.approx(sample_flight_line.lon1, abs=1e-4)

    def test_invalid_angle_type_raises(self, sample_flight_line):
        with pytest.raises(Exception):
            sample_flight_line.rotate_around_midpoint("ninety")


class TestTrack:
    def test_track_returns_linestring(self, sample_flight_line):
        from shapely.geometry import LineString
        track = sample_flight_line.track(precision=500.0)
        assert isinstance(track, LineString)
        assert track.is_valid

    def test_track_endpoints_match(self, sample_flight_line):
        track = sample_flight_line.track(precision=1000.0)
        coords = list(track.coords)
        # First point should be near (lon1, lat1)
        assert coords[0][0] == pytest.approx(sample_flight_line.lon1, abs=1e-3)
        assert coords[0][1] == pytest.approx(sample_flight_line.lat1, abs=1e-3)
        # Last point should be near (lon2, lat2)
        assert coords[-1][0] == pytest.approx(sample_flight_line.lon2, abs=1e-3)
        assert coords[-1][1] == pytest.approx(sample_flight_line.lat2, abs=1e-3)

    def test_track_point_count_scales_with_precision(self, sample_flight_line):
        fine = sample_flight_line.track(precision=100.0)
        coarse = sample_flight_line.track(precision=1000.0)
        assert len(list(fine.coords)) > len(list(coarse.coords))

    def test_track_with_quantity_precision(self, sample_flight_line):
        track = sample_flight_line.track(precision=ureg.Quantity(500, "meter"))
        assert len(list(track.coords)) > 2

    def test_track_invalid_precision_raises(self, sample_flight_line):
        with pytest.raises(Exception):
            sample_flight_line.track(precision=0.0)
        with pytest.raises(Exception):
            sample_flight_line.track(precision=-100.0)


class TestErrorPaths:
    def test_negative_altitude_raises(self):
        with pytest.raises(Exception):
            FlightLine.start_length_azimuth(
                lat1=34.0, lon1=-118.0,
                length=ureg.Quantity(10000, "meter"),
                az=90.0,
                altitude_msl=ureg.Quantity(-100, "meter"),
            )

    def test_non_quantity_length_raises(self):
        with pytest.raises(Exception):
            FlightLine.start_length_azimuth(
                lat1=34.0, lon1=-118.0,
                length=10000,  # not a Quantity
                az=90.0,
                altitude_msl=ureg.Quantity(3000, "meter"),
            )

    def test_non_numeric_azimuth_raises(self):
        with pytest.raises(Exception):
            FlightLine.start_length_azimuth(
                lat1=34.0, lon1=-118.0,
                length=ureg.Quantity(10000, "meter"),
                az="north",
                altitude_msl=ureg.Quantity(3000, "meter"),
            )

    def test_non_waypoint_init_raises(self):
        from hyplan.exceptions import HyPlanTypeError
        with pytest.raises(HyPlanTypeError):
            FlightLine(waypoint1="not_a_waypoint", waypoint2="also_not")

    def test_altitude_setter_validates(self, short_flight_line):
        with pytest.raises(Exception):
            short_flight_line.altitude_msl = ureg.Quantity(-500, "meter")

    def test_from_endpoints(self):
        fl = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0,
            lat2=34.1, lon2=-117.9,
            altitude_msl=ureg.Quantity(5000, "meter"),
            site_name="EP",
        )
        assert fl.geometry.is_valid
        assert fl.site_name == "EP"


class TestNoneAltitude:
    """FlightLine factories and serializers must handle altitude_msl=None."""

    def test_start_length_azimuth_default_altitude(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0,
            length=ureg.Quantity(10, "km"), az=90.0,
        )
        assert fl.altitude_msl is None
        assert fl.length.m_as("meter") == pytest.approx(10000, rel=0.01)

    def test_from_endpoints_default_altitude(self):
        fl = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0, lat2=34.1, lon2=-117.9,
        )
        assert fl.altitude_msl is None
        assert fl.geometry.is_valid

    def test_center_length_azimuth_default_altitude(self):
        fl = FlightLine.center_length_azimuth(
            lat=34.0, lon=-118.0,
            length=ureg.Quantity(10, "km"), az=0.0,
        )
        assert fl.altitude_msl is None
        assert fl.length.m_as("meter") == pytest.approx(10000, rel=0.01)

    def test_to_dict_none_altitude(self):
        fl = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0, lat2=34.1, lon2=-117.9,
        )
        d = fl.to_dict()
        assert d["altitude_msl"] is None

    def test_to_geojson_roundtrip_none_altitude(self):
        fl = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0, lat2=34.1, lon2=-117.9,
        )
        gj = fl.to_geojson()
        assert gj["properties"]["altitude_msl"] is None
        restored = FlightLine.from_geojson(gj)
        assert restored.altitude_msl is None
        assert restored.lat1 == pytest.approx(fl.lat1, abs=1e-4)
        assert restored.lon2 == pytest.approx(fl.lon2, abs=1e-4)

    def test_offset_north_east_preserves_none_altitude(self):
        fl = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0, lat2=34.1, lon2=-117.9,
        )
        moved = fl.offset_north_east(
            ureg.Quantity(1000, "meter"), ureg.Quantity(500, "meter"),
        )
        assert moved.altitude_msl is None
        assert moved.length.m_as("meter") == pytest.approx(
            fl.length.m_as("meter"), rel=0.01
        )


class TestFlightLineReprEq:
    def test_repr_contains_name_and_coordinates(self, sample_flight_line):
        r = repr(sample_flight_line)
        assert "Test Line" in r
        assert f"{sample_flight_line.lat1:.6f}" in r
        assert f"{sample_flight_line.lon1:.6f}" in r
        assert f"{sample_flight_line.lat2:.6f}" in r
        assert "6000 m" in r

    def test_repr_none_altitude(self):
        fl = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0, lat2=34.1, lon2=-117.9,
        )
        assert "altitude_msl=None" in repr(fl)

    def test_equal_by_value(self):
        def make():
            return FlightLine.start_length_azimuth(
                lat1=34.0, lon1=-118.0,
                length=ureg.Quantity(10, "km"), az=45.0,
                altitude_msl=ureg.Quantity(3000, "meter"),
                site_name="EQ", site_description="desc", investigator="PI",
            )
        assert make() == make()

    def test_unequal_on_field_difference(self):
        a = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0, lat2=34.1, lon2=-117.9, site_name="A",
        )
        b = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0, lat2=34.1, lon2=-117.9, site_name="B",
        )
        c = FlightLine.from_endpoints(
            lat1=34.0, lon1=-118.0, lat2=34.2, lon2=-117.9, site_name="A",
        )
        assert a != b
        assert a != c
        assert a != "not a flight line"

    def test_init_copies_caller_waypoints(self):
        """Mutating a line's altitude must not write through to the
        Waypoint objects the caller passed in."""
        from hyplan.waypoint import Waypoint

        wp1 = Waypoint(34.0, -118.0, 90.0, altitude_msl=ureg.Quantity(3000, "meter"))
        wp2 = Waypoint(34.0, -117.9, 90.0, altitude_msl=ureg.Quantity(3000, "meter"))
        fl = FlightLine(waypoint1=wp1, waypoint2=wp2, site_name="Alias")

        fl.altitude_msl = ureg.Quantity(6000, "meter")

        assert fl.altitude_msl.m_as("meter") == pytest.approx(6000)
        assert wp1.altitude_msl.m_as("meter") == pytest.approx(3000)
        assert wp2.altitude_msl.m_as("meter") == pytest.approx(3000)


class TestToGDF:
    def test_creates_geodataframe(self, sample_flight_line, short_flight_line):
        gdf = to_gdf([sample_flight_line, short_flight_line])
        assert len(gdf) == 2
        assert gdf.crs is not None


class TestAltitudeAGL:
    @pytest.fixture
    def agl_line(self, sample_flight_line):
        sample_flight_line.terrain_elevation = ureg.Quantity(1000, "meter")
        return sample_flight_line

    def test_default_has_no_terrain(self, sample_flight_line):
        assert sample_flight_line.terrain_elevation is None
        assert sample_flight_line.altitude_agl is None

    def test_exports_unchanged_without_terrain(self, sample_flight_line):
        assert "altitude_agl" not in sample_flight_line.to_dict()
        assert "altitude_agl" not in sample_flight_line.to_geojson()["properties"]
        assert "altitude_agl" not in to_gdf([sample_flight_line]).columns

    def test_altitude_agl_value(self, agl_line):
        assert agl_line.altitude_agl.m_as("meter") == pytest.approx(5000)

    def test_altitude_agl_follows_altitude_msl(self, agl_line):
        agl_line.altitude_msl = ureg.Quantity(4000, "meter")
        assert agl_line.altitude_agl.m_as("meter") == pytest.approx(3000)

    def test_altitude_agl_none_without_altitude_msl(self, agl_line):
        agl_line.altitude_msl = None
        assert agl_line.altitude_agl is None
        assert "altitude_agl" not in agl_line.to_geojson()["properties"]

    def test_terrain_elevation_converted_to_meters(self, sample_flight_line):
        sample_flight_line.terrain_elevation = ureg.Quantity(1, "km")
        assert sample_flight_line.terrain_elevation.units == ureg.meter
        assert sample_flight_line.terrain_elevation.magnitude == pytest.approx(1000)

    def test_terrain_elevation_plain_number_is_meters(self, sample_flight_line):
        sample_flight_line.terrain_elevation = 250
        assert sample_flight_line.terrain_elevation == ureg.Quantity(250, "meter")

    def test_terrain_elevation_below_sea_level(self, sample_flight_line):
        sample_flight_line.terrain_elevation = -400
        assert sample_flight_line.altitude_agl.m_as("meter") == pytest.approx(6400)

    @pytest.mark.parametrize("value", ["100", True, [100]])
    def test_terrain_elevation_invalid_type_raises(self, sample_flight_line, value):
        with pytest.raises(HyPlanTypeError):
            sample_flight_line.terrain_elevation = value

    def test_terrain_elevation_reset_to_none(self, agl_line):
        agl_line.terrain_elevation = None
        assert agl_line.altitude_agl is None

    def test_to_dict_includes_altitude_agl(self, agl_line):
        assert agl_line.to_dict()["altitude_agl"] == pytest.approx(5000)

    def test_to_geojson_includes_altitude_agl(self, agl_line):
        assert agl_line.to_geojson()["properties"]["altitude_agl"] == pytest.approx(5000)

    def test_to_gdf_includes_altitude_agl(self, agl_line, short_flight_line):
        gdf = to_gdf([agl_line, short_flight_line])
        assert gdf["altitude_agl"].iloc[0] == pytest.approx(5000)
        assert gdf["altitude_agl"].isna().iloc[1]

    def test_from_geojson_restores_terrain(self, agl_line):
        restored = FlightLine.from_geojson(agl_line.to_geojson())
        assert restored.terrain_elevation.m_as("meter") == pytest.approx(1000)
        assert restored.altitude_agl.m_as("meter") == pytest.approx(5000)

    def test_from_geojson_roundtrip_equal(self, agl_line):
        first = FlightLine.from_geojson(agl_line.to_geojson())
        second = FlightLine.from_geojson(first.to_geojson())
        assert first == second

    def test_from_geojson_without_altitude_agl(self, sample_flight_line):
        restored = FlightLine.from_geojson(sample_flight_line.to_geojson())
        assert restored.terrain_elevation is None
        assert restored.altitude_agl is None

    def test_from_geojson_ignores_agl_without_msl(self, agl_line):
        gj = agl_line.to_geojson()
        gj["properties"]["altitude_msl"] = None
        restored = FlightLine.from_geojson(gj)
        assert restored.terrain_elevation is None

    def test_eq_differs_on_terrain(self, short_flight_line):
        a = FlightLine.from_geojson(short_flight_line.to_geojson())
        b = FlightLine.from_geojson(short_flight_line.to_geojson())
        assert a == b
        a.terrain_elevation = ureg.Quantity(100, "meter")
        assert a != b
        b.terrain_elevation = ureg.Quantity(200, "meter")
        assert a != b
        b.terrain_elevation = ureg.Quantity(100, "meter")
        assert a == b

    def test_reverse_keeps_terrain(self, agl_line):
        reversed_line = agl_line.reverse()
        assert reversed_line.terrain_elevation == agl_line.terrain_elevation
        assert reversed_line.altitude_agl.m_as("meter") == pytest.approx(5000)

    def test_offset_north_east_drops_terrain(self, agl_line):
        derived = agl_line.offset_north_east(ureg.Quantity(1, "km"), ureg.Quantity(1, "km"))
        assert derived.terrain_elevation is None
        assert derived.altitude_agl is None

    def test_offset_across_drops_terrain(self, agl_line):
        assert agl_line.offset_across(ureg.Quantity(1, "km")).terrain_elevation is None

    def test_offset_along_drops_terrain(self, agl_line):
        derived = agl_line.offset_along(ureg.Quantity(1, "km"), ureg.Quantity(-1, "km"))
        assert derived.terrain_elevation is None

    def test_rotate_drops_terrain(self, agl_line):
        assert agl_line.rotate_around_midpoint(30.0).terrain_elevation is None

    def test_split_drops_terrain(self, agl_line):
        segments = agl_line.split_by_length(ureg.Quantity(10, "km"))
        assert len(segments) > 1
        assert all(seg.terrain_elevation is None for seg in segments)

    def test_partial_clip_drops_terrain(self, agl_line):
        segments = agl_line.clip_to_polygon(box(-118.3, 34.0, -118.0, 34.25))
        assert segments
        assert all(seg.terrain_elevation is None for seg in segments)

    def test_parent_unchanged_by_derived_line(self, agl_line):
        agl_line.offset_across(ureg.Quantity(1, "km"))
        agl_line.reverse().terrain_elevation = ureg.Quantity(0, "meter")
        assert agl_line.terrain_elevation == ureg.Quantity(1000, "meter")

    def test_sample_terrain(self, sample_flight_line, monkeypatch):
        calls = []

        def fake_along_track(flight_line, dem_file, precision=100.0):
            calls.append((flight_line, dem_file, precision))
            return {"min": 800.0, "mean": 1200.0, "max": 1500.0}

        monkeypatch.setattr("hyplan.terrain.terrain_elevation_along_track", fake_along_track)
        result = sample_flight_line.sample_terrain("dem.tif", precision=50.0)
        assert calls == [(sample_flight_line, "dem.tif", 50.0)]
        assert result == ureg.Quantity(1200.0, "meter")
        assert sample_flight_line.terrain_elevation == ureg.Quantity(1200.0, "meter")
        assert sample_flight_line.altitude_agl.m_as("meter") == pytest.approx(4800)


RESERVED_KEYS = [
    "geometry", "lat1", "lon1", "lat2", "lon2", "length", "altitude_msl",
    "altitude_agl", "site_name", "site_description", "investigator",
]

FIXED_DICT_KEYS = {
    "geometry", "lat1", "lon1", "lat2", "lon2", "length", "altitude_msl",
    "site_name", "site_description", "investigator",
}

FIXED_GEOJSON_KEYS = {"altitude_msl", "site_name", "site_description", "investigator"}


def _meta_line(metadata=None):
    return FlightLine.from_endpoints(
        lat1=34.0, lon1=-118.0, lat2=34.3, lon2=-117.7,
        altitude_msl=ureg.Quantity(6000, "meter"),
        site_name="Meta Line",
        metadata=metadata,
    )


class TestMetadataConstruction:
    def test_default_is_empty_dict(self, sample_flight_line):
        assert sample_flight_line.metadata == {}

    def test_init(self):
        wp1 = Waypoint(latitude=34.0, longitude=-118.0, heading=90.0,
                       altitude_msl=ureg.Quantity(3000, "meter"))
        wp2 = Waypoint(latitude=34.0, longitude=-117.9, heading=90.0,
                       altitude_msl=ureg.Quantity(3000, "meter"))
        fl = FlightLine(wp1, wp2, metadata={"campaign": "X"})
        assert fl.metadata == {"campaign": "X"}

    def test_start_length_azimuth(self):
        fl = FlightLine.start_length_azimuth(
            lat1=34.0, lon1=-118.0, length=ureg.Quantity(10, "km"), az=90.0,
            metadata={"campaign": "X"},
        )
        assert fl.metadata == {"campaign": "X"}

    def test_from_endpoints(self):
        assert _meta_line({"campaign": "X"}).metadata == {"campaign": "X"}

    def test_center_length_azimuth(self):
        fl = FlightLine.center_length_azimuth(
            lat=34.0, lon=-118.0, length=ureg.Quantity(10, "km"), az=90.0,
            metadata={"campaign": "X"},
        )
        assert fl.metadata == {"campaign": "X"}

    def test_stored_copy_is_independent_of_input(self):
        source = {"tags": ["a", "b"], "priority": 1}
        fl = _meta_line(source)
        source["tags"].append("c")
        source["priority"] = 2
        assert fl.metadata == {"tags": ["a", "b"], "priority": 1}

    def test_scalar_types_preserved(self):
        metadata = {"s": "x", "i": 1, "f": 1.5, "b": True, "n": None}
        assert _meta_line(metadata).metadata == metadata

    @pytest.mark.parametrize("key", RESERVED_KEYS)
    def test_reserved_key_raises(self, key):
        with pytest.raises(HyPlanValueError, match="reserved"):
            _meta_line({key: "x"})

    @pytest.mark.parametrize("key", [1, 1.5, ("a", "b"), None])
    def test_non_string_key_raises(self, key):
        with pytest.raises(HyPlanValueError, match="keys must be strings"):
            _meta_line({key: "x"})

    @pytest.mark.parametrize("value", [["campaign", "X"], "campaign", 1])
    def test_non_dict_raises(self, value):
        with pytest.raises(HyPlanTypeError, match="metadata must be a dict"):
            _meta_line(value)

    @pytest.mark.parametrize("value", [
        datetime.datetime(2026, 1, 1),
        object(),
        {1, 2},
        float("nan"),
        float("inf"),
        np.float64("nan"),
        [1.0, float("-inf")],
        np.array([1, 2]),
    ])
    def test_non_json_value_raises(self, value):
        with pytest.raises(HyPlanValueError, match="JSON-serializable"):
            _meta_line({"v": value})

    def test_numpy_scalars_converted(self):
        fl = _meta_line({
            "i": np.int64(3), "f32": np.float32(1.5), "f64": np.float64(2.5),
            "b": np.bool_(True), "nested": [np.int32(1)],
        })
        assert fl.metadata == {"i": 3, "f32": 1.5, "f64": 2.5, "b": True, "nested": [1]}
        assert type(fl.metadata["i"]) is int
        assert type(fl.metadata["f32"]) is float
        assert type(fl.metadata["f64"]) is float
        assert type(fl.metadata["b"]) is bool
        assert type(fl.metadata["nested"][0]) is int

    def test_tuple_stored_as_list(self):
        assert _meta_line({"pair": (1, 2)}).metadata == {"pair": [1, 2]}


class TestMetadataExport:
    def test_exports_unchanged_without_metadata(self, sample_flight_line):
        assert set(sample_flight_line.to_dict()) == FIXED_DICT_KEYS
        assert set(sample_flight_line.to_geojson()["properties"]) == FIXED_GEOJSON_KEYS

    def test_to_dict_flat(self):
        data = _meta_line({"campaign": "X", "priority": 1}).to_dict()
        assert data["campaign"] == "X"
        assert data["priority"] == 1
        assert set(data) == FIXED_DICT_KEYS | {"campaign", "priority"}

    def test_to_geojson_flat(self):
        props = _meta_line({"campaign": "X", "priority": 1}).to_geojson()["properties"]
        assert props["campaign"] == "X"
        assert props["priority"] == 1
        assert set(props) == FIXED_GEOJSON_KEYS | {"campaign", "priority"}

    def test_to_gdf_columns(self):
        gdf = to_gdf([_meta_line({"campaign": "X"}), _meta_line({"priority": 2})])
        assert gdf["campaign"].iloc[0] == "X"
        assert gdf["campaign"].isna().iloc[1]
        assert gdf["priority"].iloc[1] == 2
        assert gdf["priority"].isna().iloc[0]

    def test_with_altitude_agl(self):
        fl = _meta_line({"campaign": "X"})
        fl.terrain_elevation = ureg.Quantity(1000, "meter")
        props = fl.to_geojson()["properties"]
        assert props["altitude_agl"] == pytest.approx(5000)
        assert props["campaign"] == "X"

    def test_export_does_not_share_objects(self):
        fl = _meta_line({"tags": ["a"]})
        fl.to_geojson()["properties"]["tags"].append("b")
        fl.to_dict()["tags"].append("c")
        assert fl.metadata == {"tags": ["a"]}

    def test_edits_after_construction_exported(self):
        fl = _meta_line({"campaign": "X"})
        fl.metadata["priority"] = np.int64(2)
        del fl.metadata["campaign"]
        props = fl.to_geojson()["properties"]
        assert props["priority"] == 2
        assert "campaign" not in props

    @pytest.mark.parametrize("key, value, match", [
        ("site_name", "x", "reserved"),
        ("bad", object(), "JSON-serializable"),
        ("bad", float("nan"), "JSON-serializable"),
    ])
    def test_invalid_edit_raises_on_export(self, key, value, match):
        fl = _meta_line()
        fl.metadata[key] = value
        with pytest.raises(HyPlanValueError, match=match):
            fl.to_geojson()
        with pytest.raises(HyPlanValueError, match=match):
            fl.to_dict()
        with pytest.raises(HyPlanValueError, match=match):
            to_gdf([fl])

    def test_non_string_key_edit_raises_on_export(self):
        fl = _meta_line()
        fl.metadata[1] = "x"
        with pytest.raises(HyPlanValueError, match="keys must be strings"):
            fl.to_geojson()

    def test_acceptance_example(self):
        assert FlightLine.from_endpoints(
            34.0, -118.0, 34.3, -117.7, metadata={"campaign": "X"},
        ).to_geojson()["properties"]["campaign"] == "X"


class TestMetadataFromGeoJSON:
    def test_extra_properties_become_metadata(self):
        fl = _meta_line()
        gj = fl.to_geojson()
        gj["properties"].update({"campaign": "X", "priority": 1, "notes": None})
        restored = FlightLine.from_geojson(gj)
        assert restored.metadata == {"campaign": "X", "priority": 1, "notes": None}
        assert restored.site_name == "Meta Line"

    def test_no_extra_properties(self, sample_flight_line):
        assert FlightLine.from_geojson(sample_flight_line.to_geojson()).metadata == {}

    def test_no_properties(self):
        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[-118.0, 34.0], [-117.9, 34.0]]},
        }
        assert FlightLine.from_geojson(feature).metadata == {}

    def test_altitude_agl_not_in_metadata(self):
        fl = _meta_line({"campaign": "X"})
        fl.terrain_elevation = ureg.Quantity(1000, "meter")
        restored = FlightLine.from_geojson(fl.to_geojson())
        assert restored.metadata == {"campaign": "X"}
        assert restored.altitude_agl.m_as("meter") == pytest.approx(5000)

    def test_altitude_agl_without_msl_not_in_metadata(self):
        gj = _meta_line({"campaign": "X"}).to_geojson()
        gj["properties"]["altitude_msl"] = None
        gj["properties"]["altitude_agl"] = 5000.0
        assert FlightLine.from_geojson(gj).metadata == {"campaign": "X"}

    def test_roundtrip_equal(self):
        first = FlightLine.from_geojson(
            _meta_line({"campaign": "X", "priority": 1.25, "tags": ["a"]}).to_geojson()
        )
        second = FlightLine.from_geojson(first.to_geojson())
        assert first == second
        assert second.metadata == {"campaign": "X", "priority": 1.25, "tags": ["a"]}

    def test_pattern_keys_end_up_in_metadata(self):
        gj = _meta_line().to_geojson()
        gj["properties"].update({"line_id": "line_001", "pattern_id": "p", "pattern_kind": "k"})
        assert FlightLine.from_geojson(gj).metadata == {
            "line_id": "line_001", "pattern_id": "p", "pattern_kind": "k",
        }


class TestMetadataEquality:
    def test_eq_differs_on_metadata(self):
        base = FlightLine.from_geojson(_meta_line().to_geojson())
        a = FlightLine.from_geojson(base.to_geojson())
        b = FlightLine.from_geojson(base.to_geojson())
        assert a == b
        a.metadata["campaign"] = "X"
        assert a != b
        b.metadata["campaign"] = "Y"
        assert a != b
        b.metadata["campaign"] = "X"
        assert a == b


PROPAGATION_META = {"campaign": "X", "tags": ["a"]}


class TestMetadataPropagation:
    @pytest.mark.parametrize("derive", [
        lambda fl: fl.reverse(),
        lambda fl: fl.offset_north_east(ureg.Quantity(1, "km"), ureg.Quantity(1, "km")),
        lambda fl: fl.offset_across(ureg.Quantity(1, "km")),
        lambda fl: fl.offset_along(ureg.Quantity(1, "km"), ureg.Quantity(-1, "km")),
        lambda fl: fl.rotate_around_midpoint(30.0),
    ], ids=["reverse", "offset_north_east", "offset_across", "offset_along", "rotate"])
    def test_derived_line_keeps_independent_copy(self, derive):
        parent = _meta_line(PROPAGATION_META)
        child = derive(parent)
        assert child.metadata == PROPAGATION_META
        child.metadata["tags"].append("b")
        child.metadata["campaign"] = "Y"
        assert parent.metadata == PROPAGATION_META

    def test_split_keeps_independent_copies(self):
        parent = _meta_line(PROPAGATION_META)
        segments = parent.split_by_length(ureg.Quantity(10, "km"))
        assert len(segments) > 1
        assert all(seg.metadata == PROPAGATION_META for seg in segments)
        segments[0].metadata["tags"].append("b")
        assert parent.metadata == PROPAGATION_META
        assert segments[1].metadata == PROPAGATION_META

    def test_partial_clip_keeps_independent_copies(self):
        parent = _meta_line(PROPAGATION_META)
        segments = parent.clip_to_polygon(box(-118.1, 33.9, -117.85, 34.4))
        assert segments
        assert all(seg is not parent for seg in segments)
        assert all(seg.metadata == PROPAGATION_META for seg in segments)
        segments[0].metadata["tags"].append("b")
        assert parent.metadata == PROPAGATION_META

    def test_acceptance_value_survives(self):
        fl = FlightLine.from_endpoints(34.0, -118.0, 34.3, -117.7, metadata={"campaign": "X"})
        assert fl.reverse().to_geojson()["properties"]["campaign"] == "X"
        assert all(
            seg.to_geojson()["properties"]["campaign"] == "X"
            for seg in fl.split_by_length(ureg.Quantity(10, "km"))
        )
        clipped = fl.clip_to_polygon(box(-118.1, 33.9, -117.85, 34.4))
        assert clipped
        assert all(seg.to_geojson()["properties"]["campaign"] == "X" for seg in clipped)

    def test_derived_line_with_invalid_metadata_raises(self):
        fl = _meta_line()
        fl.metadata["bad"] = object()
        with pytest.raises(HyPlanValueError, match="JSON-serializable"):
            fl.reverse()
