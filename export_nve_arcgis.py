"""Create a styled ArcGIS Pro project and map previews from an NVE download."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# ArcPy must load its geospatial DLLs before other GIS packages on Windows.
import arcpy


def export_project(directory: Path, template: Path | None = None) -> Path:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    output = directory / "nve_weather.aprx"
    if output.exists():
        raise FileExistsError(f"ArcGIS project already exists: {output}")
    if template is None:
        template = (Path(arcpy.GetInstallInfo()["InstallDir"]) / "Resources" /
                    "ArcToolBox" / "Services" / "routingservices" / "data" / "Blank.aprx")
    if not template.is_file():
        raise FileNotFoundError(f"Specify --template with a blank ArcGIS Pro project: {template}")
    project = arcpy.mp.ArcGISProject(str(template))
    for existing in project.listMaps():
        project.deleteItem(existing)
    for existing in project.listLayouts():
        project.deleteItem(existing)
    for series in manifest["series"]:
        theme = series["theme"]
        paths = [Path(path) for path in manifest["geotiffs"]
                 if Path(path).name.startswith(f"nve_{theme}_")]
        if not paths:
            raise ValueError(f"Manifest contains no rasters for {theme}")
        map_view = project.createMap(f"NVE {theme} [{series['unit']}] - 1 km", "MAP")
        for existing in map_view.listLayers():
            map_view.removeLayer(existing)
        map_view.spatialReference = arcpy.SpatialReference(32633)
        ramp_name = "Temperature" if theme.startswith("tm") else "Blue Light to Dark"
        ramps = project.listColorRamps(ramp_name)
        if not ramps:
            raise ValueError(f"ArcGIS color ramp not found: {ramp_name}")
        for index, path in enumerate(paths):
            raster = arcpy.Raster(str(path))
            if raster.meanCellWidth != 1000 or raster.meanCellHeight != 1000:
                raise ValueError(f"Raster is not a native 1 km grid: {path}")
            arcpy.management.CalculateStatistics(str(path))
            layer = map_view.addDataFromPath(str(path))
            layer.name = f"{theme} [{series['unit']}] {path.stem.split('_')[2]}"
            layer.visible = index == 0
            symbology = layer.symbology
            symbology.updateColorizer("RasterStretchColorizer")
            symbology.colorizer.stretchType = "MinimumMaximum"
            symbology.colorizer.colorRamp = ramps[0]
            layer.symbology = symbology
            layer.saveACopy(str(path.with_suffix(".lyrx")))
            if index == 0:
                extent = raster.extent
        map_view.defaultCamera.setExtent(extent)
        layout = project.createLayout(8, 8, "INCH", f"NVE {theme} 1 km")
        frame = layout.createMapFrame(
            arcpy.Extent(0.5, 0.5, 7.5, 7.5), map_view, f"NVE {theme}",
        )
        frame.camera.setExtent(extent)
        frame.camera.scale *= 1.15
        layout.exportToPNG(str(directory / f"nve_{theme}_arcgis.png"), resolution=120)
        print(f"ArcGIS map: {map_view.name}", flush=True)
    project.saveACopy(str(output))
    check = arcpy.mp.ArcGISProject(str(output))
    if check.listBrokenDataSources():
        raise ValueError("Saved ArcGIS project has broken data sources")
    print(f"ArcGIS project: {output}", flush=True)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="NVE output directory containing manifest.json")
    parser.add_argument("--template", type=Path, help="Optional blank .aprx template")
    parser.add_argument("--open", action="store_true", help="Open the completed project in ArcGIS Pro")
    args = parser.parse_args(argv)
    project = export_project(args.directory, args.template)
    if args.open:
        os.startfile(str(project))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
