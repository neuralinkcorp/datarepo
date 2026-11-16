import { GeoJSONFeature, ParsedRecord } from "./types";

type Position = number[];

function isPosition(value: unknown): value is Position {
  if (!Array.isArray(value) || value.length < 2) return false;
  if (!value.every((coordinate) =>
    typeof coordinate === "number" && Number.isFinite(coordinate)
  )) return false;
  const [longitude, latitude] = value;
  return longitude >= -180 && longitude <= 180 &&
    latitude >= -90 && latitude <= 90;
}

function samePosition(left: Position, right: Position): boolean {
  return left[0] === right[0] && left[1] === right[1];
}

function geometryCenter(
  geometry: GeoJSONFeature["geometry"],
  problems: string[]
): [number, number] | null {
  const coordinates: unknown = geometry.coordinates;
  if (geometry.type === "Point") {
    if (!isPosition(coordinates)) {
      problems.push("Invalid point coordinates");
      return null;
    }
    if (coordinates[0] === 0 && coordinates[1] === 0) {
      problems.push("Zero/zero coordinate: verify intent");
    }
    return [coordinates[1], coordinates[0]];
  }

  if (geometry.type === "LineString") {
    if (
      !Array.isArray(coordinates) ||
      coordinates.length < 2 ||
      !coordinates.every(isPosition)
    ) {
      problems.push("Line requires at least two valid positions");
      return null;
    }
    const distinct = new Set(
      coordinates.map((position: Position) => `${position[0]},${position[1]}`)
    );
    if (distinct.size < 2) {
      problems.push("Line requires two distinct vertices");
      return null;
    }
    const middle = coordinates[Math.floor(coordinates.length / 2)] as Position;
    return [middle[1], middle[0]];
  }

  if (geometry.type === "Polygon") {
    if (!Array.isArray(coordinates) || coordinates.length === 0) {
      problems.push("Polygon requires an outer ring");
      return null;
    }
    let outer: Position[] | null = null;
    for (const [ringIndex, ring] of coordinates.entries()) {
      if (!Array.isArray(ring) || ring.length < 4 || !ring.every(isPosition)) {
        problems.push(`Polygon ring ${ringIndex + 1} has invalid coordinates`);
        return null;
      }
      const positions = ring as Position[];
      if (!samePosition(positions[0], positions[positions.length - 1])) {
        problems.push(`Polygon ring ${ringIndex + 1} is not closed`);
        return null;
      }
      const distinct = new Set(
        positions.slice(0, -1).map((position) => `${position[0]},${position[1]}`)
      );
      if (distinct.size < 3) {
        problems.push(`Polygon ring ${ringIndex + 1} needs three distinct vertices`);
        return null;
      }
      if (ringIndex === 0) outer = positions;
    }

    if (!outer) return null;
    const openRing = outer.slice(0, -1);
    const longitude = openRing.reduce((sum, point) => sum + point[0], 0) /
      openRing.length;
    const latitude = openRing.reduce((sum, point) => sum + point[1], 0) /
      openRing.length;
    return [latitude, longitude];
  }

  problems.push(`Unsupported geometry type: ${geometry.type}`);
  return null;
}

function surveyed(feature: GeoJSONFeature): boolean {
  return feature.properties?.surveyed === true;
}

export function parseRecord(feature: GeoJSONFeature): ParsedRecord {
  const problems: string[] = [];
  const props = feature.properties || {};
  const geometry = feature.geometry;
  const center = geometry
    ? geometryCenter(geometry, problems)
    : (problems.push("Missing geometry"), null);
  const valid = center !== null;

  const timestamp = props.observedAt ??
    props.observedAtEpochMs ??
    props.capturedAtEpochMs;
  let observed: Date | null = null;
  if (timestamp === undefined || timestamp === null || timestamp === "") {
    problems.push("Missing observation date");
  } else {
    const parsed = new Date(timestamp);
    if (!Number.isFinite(parsed.getTime())) {
      problems.push("Invalid observation date");
    } else {
      observed = parsed;
    }
  }

  const metadata = props.vertexMetadata;
  if (!Array.isArray(metadata) || metadata.length === 0) {
    if (geometry?.type === "Point") {
      problems.push("No vertex metadata (position source unknown)");
    }
  } else {
    metadata.forEach((vertex, index) => {
      const accuracy = vertex.accuracyMeters ?? vertex.accuracy;
      if (
        accuracy !== null &&
        accuracy !== undefined &&
        (typeof accuracy !== "number" ||
          !Number.isFinite(accuracy) ||
          accuracy < 0)
      ) {
        problems.push(`Vertex ${index + 1} has invalid horizontal accuracy`);
      }
    });
  }

  if (Array.isArray(props.normalizationWarnings)) {
    for (const warning of props.normalizationWarnings) {
      if (typeof warning === "string") problems.push(warning);
    }
  }

  const id = typeof feature.id === "string" ? feature.id : "";
  if (!id) problems.push("Missing permanent record ID");

  return {
    id,
    feature,
    geometry: {
      type: geometry?.type || "Unknown",
      valid,
      center,
      coordinates: geometry?.coordinates || [],
    },
    project: typeof props.project === "string" ? props.project : "Unknown Project",
    projectId: typeof props.projectId === "string"
      ? props.projectId
      : (typeof props.project === "string" ? props.project : "unknown"),
    type: typeof props.type === "string" ? props.type : "Unknown",
    subtype: typeof props.subtype === "string" ? props.subtype : "",
    observed,
    problems,
    isSurveyed: surveyed(feature),
  };
}

export function parseAllRecords(
  features: GeoJSONFeature[]
): { valid: ParsedRecord[]; invalid: ParsedRecord[] } {
  const valid: ParsedRecord[] = [];
  const invalid: ParsedRecord[] = [];
  for (const feature of features) {
    const record = parseRecord(feature);
    if (record.geometry.valid) valid.push(record);
    else invalid.push(record);
  }
  return { valid, invalid };
}

export function getUniqueValues(
  records: ParsedRecord[],
  key: "project" | "type"
): string[] {
  return Array.from(new Set(records.map((record) => record[key]))).sort();
}

export function formatDate(date: Date | null): string {
  if (!date || !Number.isFinite(date.getTime())) return "—";
  return date.toISOString().replace("T", " ").replace("Z", " UTC");
}

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

export function getTypeIcon(type: string): string {
  const normalized = type.toLowerCase();
  if (normalized.includes("soil")) return "🧪";
  if (normalized.includes("vegetation") || normalized.includes("tree")) return "🌳";
  if (normalized.includes("fence")) return "🔗";
  if (normalized.includes("water")) return "💧";
  if (normalized.includes("habitat")) return "🌿";
  if (normalized.includes("land use") || normalized.includes("pasture")) return "🌾";
  if (normalized.includes("structure") || normalized.includes("building")) return "🏗️";
  if (normalized.includes("marker")) return "📍";
  return "📌";
}
