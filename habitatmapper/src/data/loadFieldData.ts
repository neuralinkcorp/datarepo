import type { GeoJSONFeature } from "../types";

export interface FieldReviewIssue {
  record_id?: string;
  snapshot?: string;
  problem: string;
}

export interface FieldDataset {
  features: GeoJSONFeature[];
  review: FieldReviewIssue[];
}

export async function loadFieldData(): Promise<FieldDataset> {
  const base = import.meta.env.BASE_URL;
  const [observations, review] = await Promise.all([
    fetch(`${base}field-data/observations.geojson`, { cache: "no-store" }),
    fetch(`${base}field-data/review.json`, { cache: "no-store" }),
  ]);

  if (!observations.ok) {
    throw new Error(
      "Field dataset is unavailable. Run kwf-catalog/publish_viewer.py."
    );
  }
  if (!review.ok) {
    throw new Error("Generated field review data is unavailable.");
  }

  const document: unknown = await observations.json();
  if (
    !document ||
    typeof document !== "object" ||
    !("type" in document) ||
    document.type !== "FeatureCollection" ||
    !("features" in document) ||
    !Array.isArray(document.features)
  ) {
    throw new Error("Generated field data is not a FeatureCollection.");
  }

  const reviewDocument: unknown = await review.json();
  if (
    !Array.isArray(reviewDocument) ||
    !reviewDocument.every(
      (issue) =>
        issue !== null &&
        typeof issue === "object" &&
        "problem" in issue &&
        typeof issue.problem === "string"
    )
  ) {
    throw new Error("Generated field review data has an invalid shape.");
  }

  return {
    features: document.features as GeoJSONFeature[],
    review: reviewDocument as FieldReviewIssue[],
  };
}
