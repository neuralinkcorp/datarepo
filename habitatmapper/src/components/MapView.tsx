import { useEffect, useRef } from "react";
import L from "leaflet";
import { ParsedRecord } from "../types";
import { getGeometryColor } from "../utils";

interface MapViewProps {
  records: ParsedRecord[];
  selectedRecord: ParsedRecord | null;
  onSelectRecord: (record: ParsedRecord) => void;
  showBasemap: boolean;
}

export default function MapView({
  records,
  selectedRecord,
  onSelectRecord,
  showBasemap,
}: MapViewProps) {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const layerGroupRef = useRef<L.LayerGroup | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);

  const fitData = () => {
    const map = mapRef.current;
    if (!map || records.length === 0) return;
    const size = map.getSize();
    if (size.x <= 0 || size.y <= 0) return;

    const positions: L.LatLng[] = [];
    const collectPositions = (coordinates: unknown) => {
      if (!Array.isArray(coordinates)) return;
      if (
        coordinates.length >= 2 &&
        typeof coordinates[0] === "number" &&
        typeof coordinates[1] === "number"
      ) {
        positions.push(L.latLng(coordinates[1], coordinates[0]));
        return;
      }
      coordinates.forEach(collectPositions);
    };

    for (const record of records) {
      if (record.geometry.type === "Point" && record.geometry.center) {
        positions.push(L.latLng(...record.geometry.center));
      } else {
        collectPositions(record.geometry.coordinates);
      }
    }

    const bounds = L.latLngBounds(positions);
    if (bounds.isValid()) map.fitBounds(bounds.pad(0.1), { maxZoom: 15 });
  };

  const tooltipContent = (record: ParsedRecord): HTMLElement => {
    const content = document.createElement("span");
    content.textContent = `${record.type}${record.subtype ? ` — ${record.subtype}` : ""}`;
    return content;
  };

  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [35.16, -94.11],
      zoom: 15,
      zoomControl: true,
    });

    mapRef.current = map;
    layerGroupRef.current = L.layerGroup().addTo(map);

    const sizeTimer = window.setTimeout(() => {
      map.invalidateSize();
      fitData();
    }, 100);

    return () => {
      window.clearTimeout(sizeTimer);
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Toggle basemap
  useEffect(() => {
    if (!mapRef.current) return;

    if (showBasemap && !tileLayerRef.current) {
      tileLayerRef.current = L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
          attribution: "© OpenStreetMap contributors",
          maxZoom: 19,
        }
      ).addTo(mapRef.current);
    } else if (!showBasemap && tileLayerRef.current) {
      tileLayerRef.current.remove();
      tileLayerRef.current = null;
    }
  }, [showBasemap]);

  // Update features on map
  useEffect(() => {
    if (!mapRef.current || !layerGroupRef.current) return;

    layerGroupRef.current.clearLayers();

    for (const record of records) {
      const { geometry, feature } = record;
      const color = getGeometryColor(geometry.type);
      const isSelected = selectedRecord?.id === record.id;

      if (geometry.type === "Point" && geometry.center) {
        const marker = L.circleMarker(
          [geometry.center[0], geometry.center[1]],
          {
            radius: isSelected ? 10 : 7,
            fillColor: color,
            color: isSelected ? "#fbbf24" : "#ffffff",
            weight: isSelected ? 3 : 2,
            fillOpacity: 0.9,
          }
        );
        marker.bindTooltip(tooltipContent(record), { direction: "top" });
        marker.on("click", () => onSelectRecord(record));
        layerGroupRef.current.addLayer(marker);
      }

      if (geometry.type === "LineString") {
        const coords = geometry.coordinates as number[][];
        const latlngs = coords.map((c) => [c[1], c[0]] as [number, number]);
        const polyline = L.polyline(latlngs, {
          color: isSelected ? "#fbbf24" : color,
          weight: isSelected ? 5 : 3,
          opacity: 0.85,
        });
        polyline.bindTooltip(tooltipContent(record), { direction: "top" });
        polyline.on("click", () => onSelectRecord(record));
        layerGroupRef.current.addLayer(polyline);
      }

      if (geometry.type === "Polygon") {
        const coords = geometry.coordinates as number[][][];
        const latlngs = coords.map((ring) =>
          ring.map((coordinate) =>
            [coordinate[1], coordinate[0]] as [number, number]
          )
        );
        const polygon = L.polygon(latlngs, {
          color: isSelected ? "#fbbf24" : color,
          fillColor: color,
          fillOpacity: isSelected ? 0.4 : 0.2,
          weight: isSelected ? 3 : 2,
        });
        polygon.bindTooltip(tooltipContent(record), { direction: "top" });
        polygon.on("click", () => onSelectRecord(record));
        layerGroupRef.current.addLayer(polygon);
      }
    }

  }, [records, selectedRecord, onSelectRecord]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const frame = window.requestAnimationFrame(() => {
      map.invalidateSize();
      fitData();
    });

    return () => window.cancelAnimationFrame(frame);
  }, [records]);

  return (
    <div className="relative w-full h-full">
      <div ref={mapContainerRef} className="w-full h-full bg-slate-800" />
      {!showBasemap && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-slate-900/80 text-slate-300 text-xs px-3 py-1.5 rounded-md backdrop-blur-sm">
          Basemap off — observation shapes rendered on dark background
        </div>
      )}
      <button
        type="button"
        onClick={fitData}
        disabled={records.length === 0}
        className="absolute top-3 left-3 z-[1000] rounded bg-slate-900/90 px-3 py-1.5 text-xs text-slate-200 disabled:opacity-40"
      >
        Fit data
      </button>
      {records.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="bg-slate-900/80 text-slate-300 px-6 py-4 rounded-lg backdrop-blur-sm text-center">
            <p className="text-lg font-medium">No records to display</p>
            <p className="text-sm mt-1">Adjust filters to show data</p>
          </div>
        </div>
      )}
    </div>
  );
}
