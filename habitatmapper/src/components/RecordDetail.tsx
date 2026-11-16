import { ParsedRecord } from "../types";
import { formatDate, getGeometryColor, getTypeIcon } from "../utils";

interface RecordDetailProps {
  record: ParsedRecord;
  onClose: () => void;
}

export default function RecordDetail({ record, onClose }: RecordDetailProps) {
  const { feature, geometry, project, type, subtype, observed, isSurveyed } =
    record;
  const props = feature.properties;
  const color = getGeometryColor(geometry.type);

  return (
    <div className="p-4 space-y-4 overflow-y-auto h-full">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xl">{getTypeIcon(type)}</span>
          <div>
            <h3 className="text-sm font-semibold text-slate-100">{type}</h3>
            {subtype && (
              <p className="text-xs text-slate-400">{subtype}</p>
            )}
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-slate-500 hover:text-slate-300 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* ID badge */}
      <div className="bg-slate-800/50 rounded-lg p-2.5 border border-slate-700/50">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">ID</span>
        <p className="text-xs font-mono text-slate-300 mt-0.5">{record.id}</p>
      </div>

      {/* Project */}
      <div>
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Project</span>
        <p className="text-xs text-slate-200 mt-0.5">{project}</p>
      </div>

      {/* Geometry info */}
      <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Geometry</span>
        <div className="flex items-center gap-2 mt-1">
          <span
            className="inline-block w-3 h-3 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span className="text-xs text-slate-200">{geometry.type}</span>
          <span
            className={`text-[10px] px-1.5 py-0.5 rounded ${
              geometry.valid
                ? "bg-emerald-900/50 text-emerald-400"
                : "bg-red-900/50 text-red-400"
            }`}
          >
            {geometry.valid ? "Valid" : "Invalid"}
          </span>
        </div>
        {geometry.center && (
          <p className="text-[10px] text-slate-500 mt-1.5 font-mono">
            Center: {geometry.center[1].toFixed(6)}, {geometry.center[0].toFixed(6)}
          </p>
        )}
      </div>

      {/* Position classification */}
      <div>
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">
          Position Classification
        </span>
        <p
          className={`text-xs mt-0.5 font-medium ${
            isSurveyed ? "text-emerald-400" : "text-amber-400"
          }`}
        >
          {isSurveyed ? "✓ Marked surveyed" : "⚠ Unsurveyed"}
        </p>
      </div>

      {/* Observation date */}
      <div>
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">
          Observed (UTC)
        </span>
        <p className="text-xs text-slate-200 mt-0.5 font-mono">
          {formatDate(observed)}
        </p>
      </div>

      {/* Vertex metadata */}
      {props.vertexMetadata && Array.isArray(props.vertexMetadata) && props.vertexMetadata.length > 0 && (
        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">
            Vertex Metadata ({props.vertexMetadata.length} vertices)
          </span>
          <div className="mt-1.5 space-y-1">
            {props.vertexMetadata.map((v, i) => (
              <div
                key={i}
                className="flex items-center justify-between bg-slate-800/30 rounded px-2 py-1"
              >
                <span className="text-[10px] text-slate-400">
                  v{i}: {v.source}
                </span>
                <span className="text-[10px] font-mono text-slate-300">
                  {v.accuracy !== null && v.accuracy !== undefined
                    ? `±${v.accuracy}m`
                    : "N/A"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional properties */}
      {Object.entries(props).filter(
        ([k]) =>
          !["project", "type", "subtype", "observedAt", "vertexMetadata", "assets", "notes"].includes(k)
      ).length > 0 && (
        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">
            Additional Properties
          </span>
          <div className="mt-1.5 space-y-1">
            {Object.entries(props)
              .filter(
                ([k]) =>
                  ![
                    "project",
                    "type",
                    "subtype",
                    "observedAt",
                    "vertexMetadata",
                    "assets",
                    "notes",
                  ].includes(k)
              )
              .map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-center justify-between bg-slate-800/30 rounded px-2 py-1"
                >
                  <span className="text-[10px] text-slate-400">{key}</span>
                  <span className="text-[10px] font-mono text-slate-300">
                    {String(value)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Notes */}
      {props.notes && (
        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">
            Notes
          </span>
          <p className="text-xs text-slate-300 mt-0.5 leading-relaxed bg-slate-800/30 rounded p-2">
            {props.notes}
          </p>
        </div>
      )}

      {/* Assets */}
      {props.assets && Array.isArray(props.assets) && props.assets.length > 0 && (
        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">
            Assets ({props.assets.length})
          </span>
          <div className="mt-1.5 space-y-1">
            {props.assets.map((asset, i) => (
              <div
                key={i}
                className="bg-slate-800/30 rounded px-2 py-1.5 text-[10px] text-slate-400 font-mono"
              >
                📷 {asset.path}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* CRS info */}
      <div className="border-t border-slate-700/50 pt-3">
        <p className="text-[10px] text-slate-500 leading-relaxed">
          Original geometry and CRS are preserved. WGS 84 (EPSG:4326) is used
          for map display only. This viewer does not certify survey accuracy or
          GCP suitability.
        </p>
      </div>
    </div>
  );
}
