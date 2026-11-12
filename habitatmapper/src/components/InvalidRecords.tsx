import { ParsedRecord } from "../types";
import { formatDate, getTypeIcon } from "../utils";

interface InvalidRecordsProps {
  records: ParsedRecord[];
}

export default function InvalidRecords({ records }: InvalidRecordsProps) {
  if (records.length === 0) {
    return (
      <div className="p-4">
        <div className="bg-emerald-900/20 border border-emerald-800/50 rounded-lg p-4 text-center">
          <div className="text-emerald-400 text-2xl mb-2">✓</div>
          <p className="text-sm text-emerald-300 font-medium">
            No loading or location-validation problems found
          </p>
          <p className="text-xs text-slate-400 mt-1">
            All records have valid geometry and metadata.
          </p>
        </div>
        <p className="text-[10px] text-slate-500 mt-3 text-center">
          Passing these checks does not certify survey accuracy or GCP suitability.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-3">
      <div className="bg-amber-900/20 border border-amber-800/50 rounded-lg p-3">
        <p className="text-xs text-amber-300 font-medium">
          ⚠ {records.length} record{records.length !== 1 ? "s" : ""} excluded from the map
        </p>
        <p className="text-[10px] text-slate-400 mt-1">
          Correct their source information rather than guessing a location.
        </p>
      </div>

      <div className="space-y-2">
        {records.map((record) => (
          <div
            key={record.id}
            className="bg-slate-800/50 border border-red-900/30 rounded-lg p-3"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm">{getTypeIcon(record.type)}</span>
                <div>
                  <p className="text-xs font-medium text-slate-200">
                    {record.type}
                    {record.subtype && (
                      <span className="text-slate-400"> — {record.subtype}</span>
                    )}
                  </p>
                  <p className="text-[10px] font-mono text-slate-500">
                    {record.id}
                  </p>
                </div>
              </div>
              <span className="text-[10px] text-slate-500">
                {formatDate(record.observed)}
              </span>
            </div>

            <div className="mt-2 space-y-1">
              {record.problems.map((problem, i) => (
                <div
                  key={i}
                  className="flex items-center gap-1.5 text-[10px] text-red-400"
                >
                  <span className="w-1 h-1 rounded-full bg-red-500 flex-shrink-0" />
                  {problem}
                </div>
              ))}
            </div>

            <p className="text-[10px] text-slate-500 mt-2">
              Project: {record.project}
            </p>
          </div>
        ))}
      </div>

      <p className="text-[10px] text-slate-500 text-center pt-2 border-t border-slate-700/50">
        Passing these checks does not certify survey accuracy or GCP suitability.
      </p>
    </div>
  );
}
