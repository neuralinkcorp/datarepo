import { GeoJSONFeature, ParsedRecord } from "./types";

/**
 * Validate that a Point geometry has non-zero, non-null coordinates.
 */
function validatePointCoordinates(coords: number[]): boolean {
  if (!Array.isArray(coords) || coords.length < 2) return false;
  const [lng, lat] = coords;
  if (lng === 0 && lat === 0) return false;
  if (typeof lng !== "number" || typeof lat !== "number") return false;
  if (isNaN(lng) || isNaN(lat)) return false;
  if (lng < -180 || lng > 180 || lat < -90 || lat > 90) return false;
  return true;
}

/**
 * Compute the center of a geometry for map display.
 */
function computeCenter(
  geometry: GeoJSONFeature["geometry"]
): [number, number] | null {
  const { type, coordinates } = geometry;

  if (type === "Point") {
    const coords = coordinates as number[];
    if (validatePointCoordinates(coords)) {
      return [coords[1], coords[0]]; // [lat, lng] for Leaflet
    }
    return null;
  }

  if (type === "LineString") {
    const coords = coordinates as number[][];
    if (!coords || coords.length === 0) return null;
    const mid = Math.floor(coords.length / 2);
    return [coords[mid][1], coords[mid][0]];
  }

  if (type === "Polygon") {
    const coords = coordinates as number[][][];
    if (!coords || coords.length === 0 || !coords[0] || coords[0].length === 0)
      return null;
    // Compute centroid of first ring
    let sumLng = 0,
      sumLat = 0;
    const ring = coords[0];
    const n = ring.length - 1; // exclude closing vertex
    for (let i = 0; i < n; i++) {
      sumLng += ring[i][0];
      sumLat += ring[i][1];
    }
    return [sumLat / n, sumLng / n];
  }

  return null;
}

/**
 * Check if a record has surveyed (high-accuracy) position data.
 */
function checkIsSurveyed(feature: GeoJSONFeature): boolean {
  const meta = feature.properties?.vertexMetadata;
  if (!meta || !Array.isArray(meta) || meta.length === 0) return false;
  return meta.some(
    (v) =>
      v.accuracy !== null &&
      v.accuracy !== undefined &&
      typeof v.accuracy === "number" &&
      v.accuracy <= 5.0
  );
}

/**
 * Parse a raw GeoJSON feature into a validated record.
 */
export function parseRecord(feature: GeoJSONFeature): ParsedRecord {
  const problems: string[] = [];
  const props = feature.properties || {};
  const geom = feature.geometry;

  // Validate geometry
  let valid = true;
  let center: [number, number] | null = null;

  if (!geom || !geom.type) {
    problems.push("Missing geometry");
    valid = false;
  } else {
    center = computeCenter(geom);
    if (!center) {
      problems.push("Invalid or zero coordinates");
      valid = false;
    }
  }

  // Validate observation date
  let observed: Date | null = null;
  if (props.observedAt) {
    const d = new Date(props.observedAt);
    if (isNaN(d.getTime())) {
      problems.push("Invalid observation date");
    } else {
      observed = d;
    }
  } else {
    problems.push("Missing observation date");
  }

  // Validate vertex metadata
  const meta = props.vertexMetadata;
  if (!meta || !Array.isArray(meta) || meta.length === 0) {
    if (geom?.type === "Point") {
      problems.push("No vertex metadata (position source unknown)");
    }
  }

  const isSurveyed = checkIsSurveyed(feature);

  return {
    id: feature.id || `feature-${Math.random().toString(36).slice(2, 8)}`,
    feature,
    geometry: {
      type: geom?.type || "Unknown",
      valid,
      center,
      coordinates: geom?.coordinates || [],
    },
    project: props.project || "Unknown Project",
    type: props.type || "Unknown",
    subtype: props.subtype || "",
    observed,
    problems,
    isSurveyed,
  };
}

/**
 * Parse all features from a GeoJSON collection.
 */
export function parseAllRecords(
  features: GeoJSONFeature[]
): { valid: ParsedRecord[]; invalid: ParsedRecord[] } {
  const valid: ParsedRecord[] = [];
  const invalid: ParsedRecord[] = [];

  for (const feature of features) {
    const record = parseRecord(feature);
    if (record.geometry.valid && record.problems.length === 0) {
      valid.push(record);
    } else if (record.geometry.valid) {
      // Has valid geometry but metadata issues - still show on map
      valid.push(record);
    } else {
      invalid.push(record);
    }
  }

  return { valid, invalid };
}

/**
 * Get unique values for a property across records.
 */
export function getUniqueValues(
  records: ParsedRecord[],
  key: "project" | "type"
): string[] {
  const set = new Set<string>();
  for (const r of records) {
    set.add(r[key]);
  }
  return Array.from(set).sort();
}

/**
 * Format a date for display.
 */
export function formatDate(d: Date | null): string {
  if (!d) return "—";
  return d.toISOString().replace("T", " ").replace("Z", " UTC");
}

/**
 * Get color for a geometry type.
 */
export function getGeometryColor(type: string): string {
  switch (type) {
    case "Point":
      return "#ef4444";
    case "LineString":
      return "#3b82f6";
    case "Polygon":
      return "#22c55e";
    default:
      return "#a855f7";
  }
}

/**
 * Get icon for a record type.
 */
export function getTypeIcon(type: string): string {
  const t = type.toLowerCase();
  if (t.includes("soil")) return "🧪";
  if (t.includes("vegetation") || t.includes("tree")) return "🌳";
  if (t.includes("fence")) return "🔗";
  if (t.includes("water")) return "💧";
  if (t.includes("habitat")) return "🌿";
  if (t.includes("land use") || t.includes("pasture")) return "🌾";
  if (t.includes("structure") || t.includes("building")) return "🏗️";
  if (t.includes("marker")) return "📍";
  return "📌";
}
