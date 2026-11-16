import { FilterState } from "../types";

interface FilterPanelProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  projects: Array<{ id: string; name: string }>;
  types: string[];
  totalCount: number;
  filteredCount: number;
  onRefresh: () => void;
}

export default function FilterPanel({
  filters,
  onFiltersChange,
  projects,
  types,
  totalCount,
  filteredCount,
  onRefresh,
}: FilterPanelProps) {
  const toggleProject = (project: string) => {
    const current = filters.projects;
    const next = current.includes(project)
      ? current.filter((p) => p !== project)
      : [...current, project];
    onFiltersChange({ ...filters, projects: next });
  };

  const toggleType = (type: string) => {
    const current = filters.types;
    const next = current.includes(type)
      ? current.filter((t) => t !== type)
      : [...current, type];
    onFiltersChange({ ...filters, types: next });
  };

  const clearAll = () => {
    onFiltersChange({
      projects: [],
      types: [],
      dateFrom: "",
      dateTo: "",
    });
  };

  return (
    <div className="p-4 space-y-5 overflow-y-auto h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
          Filters
        </h2>
        <button
          onClick={clearAll}
          className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
        >
          Clear all
        </button>
      </div>

      {/* Record count */}
      <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
        <div className="flex justify-between items-center">
          <span className="text-xs text-slate-400">Showing</span>
          <span className="text-sm font-mono text-emerald-400">
            {filteredCount} / {totalCount}
          </span>
        </div>
      </div>

      {/* Projects */}
      <div>
        <label className="block text-xs font-medium text-slate-400 mb-2">
          Projects
        </label>
        <div className="space-y-1.5">
          {projects.map((project) => (
            <label
              key={project.id}
              className="flex items-center gap-2 cursor-pointer group"
            >
              <input
                type="checkbox"
                checked={filters.projects.includes(project.id)}
                onChange={() => toggleProject(project.id)}
                className="w-3.5 h-3.5 rounded border-slate-600 bg-slate-800 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0"
              />
              <span className="text-xs text-slate-300 group-hover:text-slate-100 truncate">
                {project.name}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Observation Types */}
      <div>
        <label className="block text-xs font-medium text-slate-400 mb-2">
          Observation Type
        </label>
        <div className="space-y-1.5">
          {types.map((type) => (
            <label
              key={type}
              className="flex items-center gap-2 cursor-pointer group"
            >
              <input
                type="checkbox"
                checked={filters.types.includes(type)}
                onChange={() => toggleType(type)}
                className="w-3.5 h-3.5 rounded border-slate-600 bg-slate-800 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0"
              />
              <span className="text-xs text-slate-300 group-hover:text-slate-100">
                {type}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Date Range */}
      <div>
        <label className="block text-xs font-medium text-slate-400 mb-2">
          Date Range
        </label>
        <div className="space-y-2">
          <div>
            <label className="text-[10px] text-slate-500">From</label>
            <input
              type="date"
              value={filters.dateFrom}
              onChange={(e) =>
                onFiltersChange({ ...filters, dateFrom: e.target.value })
              }
              className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="text-[10px] text-slate-500">To</label>
            <input
              type="date"
              value={filters.dateTo}
              onChange={(e) =>
                onFiltersChange({ ...filters, dateTo: e.target.value })
              }
              className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500"
            />
          </div>
        </div>
      </div>

      {/* Refresh button */}
      <button
        onClick={onRefresh}
        className="w-full bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium py-2 px-3 rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        Refresh local data
      </button>
    </div>
  );
}
