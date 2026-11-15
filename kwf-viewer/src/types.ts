export interface VertexMetadata {
  source: string;
  accuracy: number | null;
  crs?: string;
  originalCoordinates?: [number, number];
}

export interface Asset {
  path: string;
  type?: string;
}

export interface FeatureProperties {
  project: string;
  type: string;
  subtype: string;
  observedAt: string;
  notes?: string;
  soilPh?: number;
  vertexMetadata?: VertexMetadata[];
  assets?: Asset[];
  [key: string]: unknown;
}

export interface GeoJSONFeature {
  type: "Feature";
  id: string;
  geometry: {
    type: string;
    coordinates: number[] | number[][] | number[][][] | number[][][][];
  };
  properties: FeatureProperties;
}

export interface GeoJSONCollection {
  type: "FeatureCollection";
  features: GeoJSONFeature[];
}

export interface ParsedRecord {
  id: string;
  feature: GeoJSONFeature;
  geometry: {
    type: string;
    valid: boolean;
    center: [number, number] | null;
    coordinates: number[] | number[][] | number[][][] | number[][][][];
  };
  project: string;
  type: string;
  subtype: string;
  observed: Date | null;
  problems: string[];
  isSurveyed: boolean;
}

export interface FilterState {
  projects: string[];
  types: string[];
  dateFrom: string;
  dateTo: string;
}
