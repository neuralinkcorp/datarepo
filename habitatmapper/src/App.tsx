import { useState, useMemo, useCallback, useEffect } from "react";
import MapView from "./components/MapView";
import FilterPanel from "./components/FilterPanel";
import RecordDetail from "./components/RecordDetail";
import InvalidRecords from "./components/InvalidRecords";
import { demoData } from "./data/demoData";
import { loadFieldData, FieldDataset } from "./data/loadFieldData";
import { parseAllRecords, getUniqueValues } from "./utils";
import { FilterState, ParsedRecord, GeoJSONFeature } from "./types";

type RightPanel = "detail" | "review";

export default function App() {
  const [showBasemap, setShowBasemap] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<ParsedRecord | null>(null);
  const [rightPanel, setRightPanel] = useState<RightPanel>("detail");
  const [activeDataset, setActiveDataset] = useState<"field" | "demo">("field");
  const [importedFeatures, setImportedFeatures] = useState<GeoJSONFeature[] | null>(null);
  const [fieldDataset, setFieldDataset] = useState<FieldDataset | null>(null);
  const [fieldLoading, setFieldLoading] = useState(true);
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    projects: [],
    types: [],
    dateFrom: "",
    dateTo: "",
  });
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let active = true;
    setFieldLoading(true);
    setFieldError(null);
    setFieldDataset(null);
    setSelectedRecord(null);
    loadFieldData()
      .then((dataset) => {
        if (active) setFieldDataset(dataset);
      })
      .catch((error: unknown) => {
        if (active) {
          setFieldError(
            error instanceof Error ? error.message : "Unable to load field data."
          );
        }
      })
      .finally(() => {
        if (active) setFieldLoading(false);
      });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  // Active features source
  const currentFeatures = useMemo(() => {
    if (importedFeatures) return importedFeatures;
    if (activeDataset === "demo") return demoData.features;
    return fieldDataset?.features ?? [];
  }, [importedFeatures, activeDataset, fieldDataset]);

  // Parse all records from active data
  const { valid: allRecords, invalid: invalidRecords } = useMemo(() => {
    return parseAllRecords(currentFeatures);
  }, [currentFeatures]);

  // Get unique filter values
  const projects = useMemo(() => {
    const labels = new Map<string, string>();
    for (const record of allRecords) {
      if (!labels.has(record.projectId)) labels.set(record.projectId, record.project);
    }
    return Array.from(labels, ([id, name]) => ({ id, name }))
      .sort((left, right) => left.name.localeCompare(right.name));
  }, [allRecords]);
  const types = useMemo(() => getUniqueValues(allRecords, "type"), [allRecords]);

  // Apply filters
  const filteredRecords = useMemo(() => {
    return allRecords.filter((record) => {
      // Project filter
      if (filters.projects.length > 0 && !filters.projects.includes(record.projectId)) {
        return false;
      }
      // Type filter
      if (filters.types.length > 0 && !filters.types.includes(record.type)) {
        return false;
      }
      // Date range filter
      if (filters.dateFrom && record.observed) {
        const from = new Date(filters.dateFrom);
        if (record.observed < from) return false;
      }
      if (filters.dateTo && record.observed) {
        const to = new Date(filters.dateTo);
        to.setHours(23, 59, 59, 999);
        if (record.observed > to) return false;
      }
      return true;
    });
  }, [allRecords, filters]);

  const reviewRecords = useMemo(
    () => [
      ...invalidRecords,
      ...allRecords.filter((record) => record.problems.length > 0),
    ],
    [allRecords, invalidRecords]
  );
  const pythonReviewIssues =
    activeDataset === "field" && !importedFeatures
      ? fieldDataset?.review ?? []
      : [];

  useEffect(() => {
    if (
      selectedRecord &&
      !filteredRecords.some((record) => record.id === selectedRecord.id)
    ) {
      setSelectedRecord(null);
    }
  }, [filteredRecords, selectedRecord]);

  const handleSelectRecord = useCallback((record: ParsedRecord) => {
    setSelectedRecord(record);
    setRightPanel("detail");
  }, []);

  const handleExportGeoJSON = useCallback(() => {
    const exportData = {
      type: "FeatureCollection" as const,
      features: filteredRecords.map((r) => r.feature),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/geo+json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "field-atlas-filtered.geojson";
    a.click();
    URL.revokeObjectURL(url);
  }, [filteredRecords]);

  const handleImportGeoJSON = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const text = e.target?.result as string;
          const parsed = JSON.parse(text);
          if (parsed.type === "FeatureCollection" && Array.isArray(parsed.features)) {
            setSelectedRecord(null);
            setFilters({ projects: [], types: [], dateFrom: "", dateTo: "" });
            setImportedFeatures(parsed.features);
          } else if (parsed.type === "Feature") {
            setSelectedRecord(null);
            setFilters({ projects: [], types: [], dateFrom: "", dateTo: "" });
            setImportedFeatures([parsed]);
          } else {
            alert("File does not appear to be valid GeoJSON.");
          }
        } catch (err) {
          console.error("Failed to parse GeoJSON", err);
          alert("Error parsing GeoJSON file.");
        }
      };
      reader.readAsText(file);
    },
    []
  );

  const handleRefresh = useCallback(() => {
    setSelectedRecord(null);
    setRefreshKey((k) => k + 1);
  }, []);

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-900 text-slate-100 overflow-hidden">
      {/* Top bar */}
      <header className="h-12 bg-slate-950 border-b border-slate-800 flex items-center justify-between px-4 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-emerald-500 rounded flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
              </svg>
            </div>
            <h1 className="text-sm font-semibold text-slate-100">KwF Field Data Viewer</h1>
          </div>
          <span className="text-[10px] bg-amber-900/30 text-amber-400 px-2 py-0.5 rounded border border-amber-800/50">
            Read-only visual inspection
          </span>

          {/* Dataset Switcher */}
          <div className="flex items-center bg-slate-900 rounded p-0.5 border border-slate-800 text-xs ml-2">
            <button
              onClick={() => {
                setImportedFeatures(null);
                setSelectedRecord(null);
                setFilters({ projects: [], types: [], dateFrom: "", dateTo: "" });
                setActiveDataset("field");
              }}
              className={`px-2.5 py-1 rounded transition-colors ${
                !importedFeatures && activeDataset === "field"
                  ? "bg-emerald-600 text-white font-medium"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Field Data ({fieldDataset?.features.length ?? "…"})
            </button>
            <button
              onClick={() => {
                setImportedFeatures(null);
                setSelectedRecord(null);
                setFilters({ projects: [], types: [], dateFrom: "", dateTo: "" });
                setActiveDataset("demo");
              }}
              className={`px-2.5 py-1 rounded transition-colors ${
                !importedFeatures && activeDataset === "demo"
                  ? "bg-emerald-600 text-white font-medium"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Demo Data
            </button>
            {importedFeatures && (
              <span className="bg-sky-600 text-white px-2.5 py-1 rounded text-xs">
                Imported ({importedFeatures.length})
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Basemap toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showBasemap}
              onChange={(e) => setShowBasemap(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-slate-600 bg-slate-800 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0"
            />
            <span className="text-xs text-slate-400">OSM basemap</span>
          </label>

          {/* Import GeoJSON button */}
          <label className="text-xs bg-slate-800 hover:bg-slate-700 cursor-pointer text-slate-200 px-3 py-1.5 rounded border border-slate-700 transition-colors flex items-center gap-1.5">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Import GeoJSON
            <input
              type="file"
              accept=".geojson,.json"
              onChange={handleImportGeoJSON}
              className="hidden"
            />
          </label>

          {/* Export button */}
          <button
            onClick={handleExportGeoJSON}
            disabled={filteredRecords.length === 0}
            className="text-xs bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-200 px-3 py-1.5 rounded border border-slate-700 transition-colors flex items-center gap-1.5"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export GeoJSON
          </button>

          {/* Review tab */}
          <button
            onClick={() => setRightPanel("review")}
            className={`text-xs px-3 py-1.5 rounded border transition-colors flex items-center gap-1.5 ${
              rightPanel === "review"
                ? "bg-amber-900/30 border-amber-700/50 text-amber-300"
                : "bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200"
            }`}
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            Needs Review ({reviewRecords.length + pythonReviewIssues.length})
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar - Filters */}
        <aside className="w-64 bg-slate-900 border-r border-slate-800 flex-shrink-0 overflow-hidden flex flex-col">
          <FilterPanel
            filters={filters}
            onFiltersChange={setFilters}
            projects={projects}
            types={types}
            totalCount={allRecords.length}
            filteredCount={filteredRecords.length}
            onRefresh={handleRefresh}
          />
        </aside>

        {/* Map area */}
        <main className="flex-1 relative">
          <MapView
            records={filteredRecords}
            selectedRecord={selectedRecord}
            onSelectRecord={handleSelectRecord}
            showBasemap={showBasemap}
          />

          {activeDataset === "field" && !importedFeatures && fieldLoading && (
            <div className="absolute top-3 right-3 rounded bg-slate-950/90 px-3 py-2 text-xs text-slate-200">
              Loading generated field data…
            </div>
          )}
          {activeDataset === "field" && !importedFeatures && fieldError && (
            <div role="alert" className="absolute top-3 right-3 max-w-md rounded border border-red-800 bg-slate-950/95 px-3 py-2 text-xs text-red-300">
              {fieldError}
            </div>
          )}

          {/* Legend */}
          <div className="absolute bottom-4 left-4 bg-slate-900/90 backdrop-blur-sm rounded-lg p-3 border border-slate-700/50">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Legend</p>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-red-500" />
                <span className="text-[10px] text-slate-300">Points</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 h-0.5 bg-blue-500 rounded" />
                <span className="text-[10px] text-slate-300">Lines</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm bg-green-500/40 border border-green-500" />
                <span className="text-[10px] text-slate-300">Polygons</span>
              </div>
            </div>
          </div>
        </main>

        {/* Right sidebar - Detail / Review */}
        <aside className="w-80 bg-slate-900 border-l border-slate-800 flex-shrink-0 overflow-hidden flex flex-col">
          {/* Tab switcher */}
          <div className="flex border-b border-slate-800">
            <button
              onClick={() => setRightPanel("detail")}
              className={`flex-1 text-xs py-2.5 px-3 transition-colors ${
                rightPanel === "detail"
                  ? "text-emerald-400 border-b-2 border-emerald-500 bg-slate-800/30"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              Record Detail
            </button>
            <button
              onClick={() => setRightPanel("review")}
              className={`flex-1 text-xs py-2.5 px-3 transition-colors ${
                rightPanel === "review"
                  ? "text-amber-400 border-b-2 border-amber-500 bg-slate-800/30"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              Needs Review
            </button>
          </div>

          {/* Panel content */}
          <div className="flex-1 overflow-hidden">
            {rightPanel === "detail" ? (
              selectedRecord ? (
                <RecordDetail
                  record={selectedRecord}
                  onClose={() => setSelectedRecord(null)}
                />
              ) : (
                <div className="p-4 flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="text-slate-600 text-3xl mb-3">🗺️</div>
                    <p className="text-xs text-slate-500">
                      Click a feature on the map to view its details
                    </p>
                  </div>
                </div>
              )
            ) : (
              <InvalidRecords
                records={reviewRecords}
                reviewIssues={pythonReviewIssues}
              />
            )}
          </div>
        </aside>
      </div>

      {/* Footer status bar */}
      <footer className="h-7 bg-slate-950 border-t border-slate-800 flex items-center justify-between px-4 flex-shrink-0">
        <div className="flex items-center gap-4">
          <span className="text-[10px] text-slate-500">
            Source: {importedFeatures
              ? "Imported GeoJSON"
              : activeDataset === "demo"
                ? "Synthetic demo dataset (explicit selection)"
                : "Generated normalized field snapshots"}
          </span>
          <span className="text-[10px] text-slate-600">|</span>
          <span className="text-[10px] text-slate-500">
            CRS: WGS 84 (EPSG:4326) for display
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[10px] text-slate-500">
            {filteredRecords.length} valid · {reviewRecords.length + pythonReviewIssues.length} to review
          </span>
          <span className="text-[10px] text-slate-600">|</span>
          <span className="text-[10px] text-emerald-500/70">Read-only viewer</span>
        </div>
      </footer>
    </div>
  );
}
