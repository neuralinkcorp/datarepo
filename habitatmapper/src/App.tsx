import { useState, useMemo, useCallback } from "react";
import MapView from "./components/MapView";
import FilterPanel from "./components/FilterPanel";
import RecordDetail from "./components/RecordDetail";
import InvalidRecords from "./components/InvalidRecords";
import { demoData } from "./data/demoData";
import { parseAllRecords, getUniqueValues } from "./utils";
import { FilterState, ParsedRecord } from "./types";

type RightPanel = "detail" | "review";

export default function App() {
  const [showBasemap, setShowBasemap] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<ParsedRecord | null>(null);
  const [rightPanel, setRightPanel] = useState<RightPanel>("detail");
  const [filters, setFilters] = useState<FilterState>({
    projects: [],
    types: [],
    dateFrom: "",
    dateTo: "",
  });
  const [refreshKey, setRefreshKey] = useState(0);

  // Parse all records from demo data
  const { valid: allRecords, invalid: invalidRecords } = useMemo(() => {
    return parseAllRecords(demoData.features);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  // Get unique filter values
  const projects = useMemo(() => getUniqueValues(allRecords, "project"), [allRecords]);
  const types = useMemo(() => getUniqueValues(allRecords, "type"), [allRecords]);

  // Apply filters
  const filteredRecords = useMemo(() => {
    return allRecords.filter((record) => {
      // Project filter
      if (filters.projects.length > 0 && !filters.projects.includes(record.project)) {
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

  const handleRefresh = useCallback(() => {
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
            Needs Review ({invalidRecords.length})
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
              <InvalidRecords records={invalidRecords} />
            )}
          </div>
        </aside>
      </div>

      {/* Footer status bar */}
      <footer className="h-7 bg-slate-950 border-t border-slate-800 flex items-center justify-between px-4 flex-shrink-0">
        <div className="flex items-center gap-4">
          <span className="text-[10px] text-slate-500">
            Source: Synthetic demo dataset (kwf-field-data/captures)
          </span>
          <span className="text-[10px] text-slate-600">|</span>
          <span className="text-[10px] text-slate-500">
            CRS: WGS 84 (EPSG:4326) for display
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[10px] text-slate-500">
            {filteredRecords.length} valid · {invalidRecords.length} invalid
          </span>
          <span className="text-[10px] text-slate-600">|</span>
          <span className="text-[10px] text-emerald-500/70">Read-only viewer</span>
        </div>
      </footer>
    </div>
  );
}
