import { ParsedRecord } from "../types";
import { FieldReviewIssue } from "../data/loadFieldData";
import { formatDate, getTypeIcon } from "../utils";

interface InvalidRecordsProps {
  records: ParsedRecord[];
  reviewIssues?: FieldReviewIssue[];
}

export default function InvalidRecords({
  records,
  reviewIssues = [],
}: InvalidRecordsProps) {
  if (records.length === 0 && reviewIssues.length === 0) {
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
          ⚠ {records.length + reviewIssues.length} item{records.length + reviewIssues.length !== 1 ? "s" : ""} to review
        </p>
        <p className="text-[10px] text-slate-400 mt-1">
          Geometry warnings may remain visible on the map; correct source data rather than guessing.
        </p>
      </div>

      {reviewIssues.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-[10px] uppercase tracking-wider text-amber-300">
            Python normalization issues
          </h3>
          {reviewIssues.map((issue, index) => (
            <div
              key={`${issue.snapshot ?? "snapshot"}-${issue.record_id ?? index}-${index}`}
              className="bg-slate-800/50 border border-amber-900/30 rounded-lg p-3"
            >
              {issue.record_id && (
                <p className="text-[10px] font-mono text-slate-400">
                  {issue.record_id}
                </p>
              )}
              {issue.snapshot && (
                <p className="text-[10px] text-slate-500 break-all">
                  {issue.snapshot}
                </p>
              )}
              <p className="text-[10px] text-amber-300 mt-1">
                {issue.problem}
              </p>
            </div>
          ))}
        </div>
      )}

      <div className="space-y-2">
        {records.map((record) => (
          <div
            key={record.id}
            className={`bg-slate-800/50 rounded-lg p-3 ${
              record.geometry.valid
                ? "border border-amber-900/30"
                : "border border-red-900/30"
            }`}
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
                  className="flex items-center gap-1.5 text-[10px] text-amber-300"
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
