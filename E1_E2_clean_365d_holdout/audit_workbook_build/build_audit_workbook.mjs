import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = String.raw`F:\nbeatsx-lstm\E1_E2_clean_365d_holdout`;
const outputDir = path.join(root, "outputs", "e1_e2_audit");
const previewDir = path.join(root, "audit_workbook_build", "previews");
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const imports = [
  ["Config selection", "nbl_validation_metrics_seed_summary.csv"],
  ["NBL test variants", "nbl_variants_vs_persistence_365d.csv"],
  ["Five-model metrics", "metrics_365d_selected_config_seed_summary.csv"],
  ["E2 seed summary", "error_by_change_tercile_selected_config_seed_summary.csv"],
  ["Forecasts 365d", "forecasts_365d_selected_config.csv"],
  ["Forecasts all seeds", "forecasts_365d_selected_config_all_seeds.csv"],
  ["Boundary audit", "training_boundary_audit_selected_config.csv"],
];

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i += 1;
      row.push(field);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.map((values, rowIndex) => values.map((value) => {
    if (rowIndex === 0 || value === "") return value;
    if (value === "True") return true;
    if (value === "False") return false;
    const number = Number(value);
    return Number.isFinite(number) ? number : value;
  }));
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README");

for (const [sheetName, fileName] of imports) {
  const csv = await fs.readFile(path.join(root, fileName), "utf8");
  const rows = parseCsv(csv);
  const sheet = workbook.worksheets.add(sheetName);
  sheet.getRangeByIndexes(0, 0, rows.length, rows[0].length).values = rows;
}

readme.getRange("A1").values = [["E1/E2 clean 365-day holdout audit"]];
readme.getRange("A3:B10").values = [
  ["Protocol", "Final 365 days excluded from fitting, scaler fitting, early stopping, and configuration selection"],
  ["Training end", "2021-05-24"],
  ["Validation", "2021-05-25 to 2023-05-24 (730 days)"],
  ["Test", "2023-05-25 to 2024-05-23 (365 days)"],
  ["Fixed seeds", "1, 42, 2024"],
  ["Selection rule", "Lowest mean validation RMSE across all three seeds"],
  ["Selected variant", null],
  ["Primary conclusion", "The validation-selected configuration does not beat Persistence on mean test RMSE across seeds."],
];
readme.getRange("B9").formulas = [["='Config selection'!A3"]];
readme.getRange("A12:F12").values = [["Model", "MAE mean", "MSE mean", "RMSE mean", "sMAPE mean", "R2 mean"]];
readme.getRange("A13:A14").values = [["NBeatsx_LSTM (selected)"], ["Persistence"]];
readme.getRange("B13:F13").formulas = [[
  "='Five-model metrics'!C4",
  "='Five-model metrics'!E4",
  "='Five-model metrics'!G4",
  "='Five-model metrics'!I4",
  "='Five-model metrics'!K4",
]];
readme.getRange("B14:F14").formulas = [[
  "='Five-model metrics'!C5",
  "='Five-model metrics'!E5",
  "='Five-model metrics'!G5",
  "='Five-model metrics'!I5",
  "='Five-model metrics'!K5",
]];
readme.getRange("A16:B21").values = [
  ["Interpretation", "Primary statistics are per-seed metrics summarized as mean and standard deviation; the mean-forecast tidy file is secondary."],
  ["E2 quiet", "NBeatsx_LSTM is worse than Persistence."],
  ["E2 medium", "Mean RMSE ratio is slightly below 1, but one of three seeds is above 1."],
  ["E2 volatile", "Mean RMSE ratio is slightly below 1, but one of three seeds is above 1."],
  ["Legacy Table VII", "Not reproduced by this clean experiment because the split, preprocessing leakage control, and date alignment differ."],
  ["Evidence root", root],
];

readme.showGridLines = false;
readme.freezePanes.freezeRows(1);
readme.getRange("A1:H1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A3:A10").format = { fill: "#D9EAF7", font: { bold: true } };
readme.getRange("A12:F12").format = { fill: "#5B9BD5", font: { bold: true, color: "#FFFFFF" } };
readme.getRange("A16:A21").format = { fill: "#E2F0D9", font: { bold: true } };
readme.getRange("A1:H21").format.verticalAlignment = "center";
readme.getRange("A3:B21").format.wrapText = true;
readme.getRange("A1:H21").format.borders = { preset: "insideHorizontal", style: "thin", color: "#D9E2F3" };
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 80;
readme.getRange("C:F").format.columnWidth = 16;
readme.getRange("A1:H21").format.autofitRows();
readme.getRange("A1:H1").format.rowHeight = 30;
readme.getRange("B13:F14").format.numberFormat = "0.0000";

for (const [sheetName] of imports) {
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange();
  const header = used.getRow(0);
  header.format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
  used.format.verticalAlignment = "center";
  used.format.autofitColumns();
  header.format.rowHeight = 30;
}

workbook.worksheets.getItem("Config selection").getRange("B2:K4").format.numberFormat = "0.0000";
workbook.worksheets.getItem("NBL test variants").getRange("B2:G4").format.numberFormat = "0.0000";
workbook.worksheets.getItem("Five-model metrics").getRange("C2:L6").format.numberFormat = "0.0000";
workbook.worksheets.getItem("E2 seed summary").getRange("D2:M16").format.numberFormat = "0.0000";
workbook.worksheets.getItem("E2 seed summary").getRange("D2:E16").format.numberFormat = "0";
workbook.worksheets.getItem("Forecasts 365d").getRange("C2:D1826").format.numberFormat = "0.0000000000";
workbook.worksheets.getItem("Forecasts all seeds").getRange("E2:F4746").format.numberFormat = "0.0000000000";

workbook.worksheets.getItem("Config selection").getRange("A:A").format.columnWidth = 28;
workbook.worksheets.getItem("Config selection").getRange("B:K").format.columnWidth = 14;
workbook.worksheets.getItem("NBL test variants").getRange("A:A").format.columnWidth = 28;
workbook.worksheets.getItem("NBL test variants").getRange("B:F").format.columnWidth = 16;
workbook.worksheets.getItem("NBL test variants").getRange("G:G").format.columnWidth = 28;
workbook.worksheets.getItem("NBL test variants").getRange("H:H").format.columnWidth = 20;
workbook.worksheets.getItem("Five-model metrics").getRange("A:A").format.columnWidth = 18;
workbook.worksheets.getItem("Five-model metrics").getRange("B:B").format.columnWidth = 28;
workbook.worksheets.getItem("Five-model metrics").getRange("C:L").format.columnWidth = 14;
workbook.worksheets.getItem("E2 seed summary").getRange("A:B").format.columnWidth = 14;
workbook.worksheets.getItem("E2 seed summary").getRange("C:C").format.columnWidth = 26;
workbook.worksheets.getItem("E2 seed summary").getRange("D:I").format.columnWidth = 14;
workbook.worksheets.getItem("E2 seed summary").getRange("J:M").format.columnWidth = 27;
workbook.worksheets.getItem("Forecasts 365d").getRange("A:A").format.columnWidth = 14;
workbook.worksheets.getItem("Forecasts 365d").getRange("B:B").format.columnWidth = 18;
workbook.worksheets.getItem("Forecasts 365d").getRange("C:D").format.columnWidth = 20;
workbook.worksheets.getItem("Forecasts all seeds").getRange("A:A").format.columnWidth = 14;
workbook.worksheets.getItem("Forecasts all seeds").getRange("B:B").format.columnWidth = 18;
workbook.worksheets.getItem("Forecasts all seeds").getRange("C:C").format.columnWidth = 28;
workbook.worksheets.getItem("Forecasts all seeds").getRange("D:D").format.columnWidth = 10;
workbook.worksheets.getItem("Forecasts all seeds").getRange("E:F").format.columnWidth = 20;
workbook.worksheets.getItem("Forecasts all seeds").getRange("G:G").format.columnWidth = 10;
workbook.worksheets.getItem("Boundary audit").getRange("A:A").format.columnWidth = 42;
workbook.worksheets.getItem("Boundary audit").getRange("B:B").format.columnWidth = 18;
workbook.worksheets.getItem("Boundary audit").getRange("C:C").format.columnWidth = 28;
workbook.worksheets.getItem("Boundary audit").getRange("D:D").format.columnWidth = 10;
workbook.worksheets.getItem("Boundary audit").getRange("E:G").format.columnWidth = 17;
workbook.worksheets.getItem("Boundary audit").getRange("H:H").format.columnWidth = 12;
workbook.worksheets.getItem("Boundary audit").getRange("I:I").format.columnWidth = 30;
workbook.worksheets.getItem("Boundary audit").getRange("J:J").format.columnWidth = 90;
for (const [sheetName] of imports) {
  workbook.worksheets.getItem(sheetName).getUsedRange().getRow(0).format.rowHeight = 42;
}

const keyInspect = await workbook.inspect({
  kind: "table",
  range: "README!A1:F21",
  include: "values,formulas",
  tableMaxRows: 24,
  tableMaxCols: 8,
});
await fs.writeFile(path.join(outputDir, "workbook_key_inspection.ndjson"), keyInspect.ndjson, "utf8");

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(outputDir, "workbook_formula_error_scan.ndjson"), errors.ndjson, "utf8");

const previewRanges = {
  README: "A1:F21",
  "Config selection": "A1:K4",
  "NBL test variants": "A1:H4",
  "Five-model metrics": "A1:N6",
  "E2 seed summary": "A1:M16",
  "Forecasts 365d": "A1:D20",
  "Forecasts all seeds": "A1:H20",
  "Boundary audit": "A1:J13",
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const preview = await workbook.render({ sheetName, range, scale: 1.5, format: "png" });
  await fs.writeFile(
    path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, "E1_E2_clean_365d_audit.xlsx"));
