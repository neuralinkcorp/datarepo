import { GeoJSONCollection } from "../types";

export const demoData: GeoJSONCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      id: "demo-point-001",
      geometry: {
        type: "Point",
        coordinates: [-94.11, 35.16],
      },
      properties: {
        project: "Arkansas Viewer Demo — Synthetic",
        type: "Soil",
        subtype: "Sample",
        observedAt: "2026-10-29T15:00:00Z",
        soilPh: 6.4,
        notes: "Synthetic test record. Not a measured field position.",
        vertexMetadata: [
          {
            source: "manual",
            accuracy: null,
          },
        ],
        assets: [],
      },
    },
    {
      type: "Feature",
      id: "demo-point-002",
      geometry: {
        type: "Point",
        coordinates: [-94.105, 35.162],
      },
      properties: {
        project: "Arkansas Viewer Demo — Synthetic",
        type: "Soil",
        subtype: "Sample",
        observedAt: "2026-10-29T15:05:00Z",
        soilPh: 5.8,
        notes: "Second synthetic soil sample point.",
        vertexMetadata: [
          {
            source: "gps",
            accuracy: 3.2,
            crs: "EPSG:4326",
          },
        ],
        assets: [],
      },
    },
    {
      type: "Feature",
      id: "demo-point-003",
      geometry: {
        type: "Point",
        coordinates: [-94.115, 35.157],
      },
      properties: {
        project: "Ozark Survey 2026",
        type: "Vegetation",
        subtype: "Tree",
        observedAt: "2026-10-30T09:30:00Z",
        notes: "Large oak specimen at field edge. GPS surveyed with RTK.",
        vertexMetadata: [
          {
            source: "rtk-gps",
            accuracy: 0.02,
            crs: "EPSG:4326",
          },
        ],
        assets: [],
      },
    },
    {
      type: "Feature",
      id: "demo-line-001",
      geometry: {
        type: "LineString",
        coordinates: [
          [-94.11, 35.16],
          [-94.108, 35.161],
          [-94.106, 35.161],
        ],
      },
      properties: {
        project: "Arkansas Viewer Demo — Synthetic",
        type: "Fence",
        subtype: "Fence line",
        observedAt: "2026-10-29T15:10:00Z",
        notes: "Synthetic fence used to test line rendering.",
        vertexMetadata: [
          { source: "manual", accuracy: null },
          { source: "manual", accuracy: null },
          { source: "manual", accuracy: null },
        ],
        assets: [],
      },
    },
    {
      type: "Feature",
      id: "demo-line-002",
      geometry: {
        type: "LineString",
        coordinates: [
          [-94.112, 35.159],
          [-94.11, 35.159],
          [-94.108, 35.1595],
          [-94.106, 35.16],
        ],
      },
      properties: {
        project: "Ozark Survey 2026",
        type: "Waterway",
        subtype: "Stream",
        observedAt: "2026-10-30T10:00:00Z",
        notes: "Seasonal creek bed. Flow observed during survey.",
        vertexMetadata: [
          { source: "gps", accuracy: 2.5, crs: "EPSG:4326" },
          { source: "gps", accuracy: 2.1, crs: "EPSG:4326" },
          { source: "gps", accuracy: 1.8, crs: "EPSG:4326" },
          { source: "gps", accuracy: 2.3, crs: "EPSG:4326" },
        ],
        assets: [],
      },
    },
    {
      type: "Feature",
      id: "demo-area-001",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [-94.112, 35.158],
            [-94.108, 35.158],
            [-94.108, 35.16],
            [-94.112, 35.16],
            [-94.112, 35.158],
          ],
        ],
      },
      properties: {
        project: "Arkansas Viewer Demo — Synthetic",
        type: "Habitat",
        subtype: "Survey area",
        observedAt: "2026-10-29T15:20:00Z",
        notes: "Synthetic area. Not a property boundary.",
        vertexMetadata: [
          { source: "manual", accuracy: null },
          { source: "manual", accuracy: null },
          { source: "manual", accuracy: null },
          { source: "manual", accuracy: null },
          { source: "manual", accuracy: null },
        ],
        assets: [],
      },
    },
    {
      type: "Feature",
      id: "demo-area-002",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [-94.116, 35.155],
            [-94.113, 35.155],
            [-94.113, 35.157],
            [-94.116, 35.157],
            [-94.116, 35.155],
          ],
        ],
      },
      properties: {
        project: "Ozark Survey 2026",
        type: "Land Use",
        subtype: "Pasture",
        observedAt: "2026-10-30T11:00:00Z",
        notes: "Grazed pasture, mixed grass. RTK surveyed boundary.",
        vertexMetadata: [
          { source: "rtk-gps", accuracy: 0.01, crs: "EPSG:4326" },
          { source: "rtk-gps", accuracy: 0.01, crs: "EPSG:4326" },
          { source: "rtk-gps", accuracy: 0.01, crs: "EPSG:4326" },
          { source: "rtk-gps", accuracy: 0.01, crs: "EPSG:4326" },
          { source: "rtk-gps", accuracy: 0.01, crs: "EPSG:4326" },
        ],
        assets: [],
      },
    },
    // Invalid records - missing coordinates
    {
      type: "Feature",
      id: "demo-invalid-001",
      geometry: {
        type: "Point",
        coordinates: [0, 0],
      },
      properties: {
        project: "Arkansas Viewer Demo — Synthetic",
        type: "Structure",
        subtype: "Building",
        observedAt: "2026-10-29T16:00:00Z",
        notes: "INVALID: Coordinates are null/zero. No valid position data.",
        vertexMetadata: [
          {
            source: "unknown",
            accuracy: null,
          },
        ],
        assets: [],
      },
    },
    // Another invalid - no observed date
    {
      type: "Feature",
      id: "demo-invalid-002",
      geometry: {
        type: "Point",
        coordinates: [-94.1, 35.155],
      },
      properties: {
        project: "Arkansas Viewer Demo — Synthetic",
        type: "Marker",
        subtype: "Unknown",
        observedAt: "",
        notes: "INVALID: Missing observation date and vertex metadata.",
        assets: [],
      },
    },
  ],
};
