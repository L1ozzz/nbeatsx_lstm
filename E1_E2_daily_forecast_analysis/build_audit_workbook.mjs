import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "file:///C:/Users/dell/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const here = "F:/nbeatsx-lstm/E1_E2_daily_forecast_analysis";
const imports = [
  ["forecasts_28d.csv", "Forecasts28"],
  ["metrics_28d.csv", "Metrics28"],
  ["table_vii_reproduction_checks.csv", "TableVII_Checks"],
  ["error_by_change_tercile.csv", "E2_28"],
  ["forecasts_365d_partial.csv", "Forecasts365_Partial"],
  ["metrics_365d_partial.csv", "Metrics365_Partial"],
  ["error_by_change_tercile_365d_partial.csv", "E2_365_Partial"],
];

const firstText = await fs.readFile(path.join(here, imports[0][0]), "utf8");
const workbook = await Workbook.fromCSV(firstText, { sheetName: imports[0][1] });
for (const [filename, sheetName] of imports.slice(1)) {
  const csvText = await fs.readFile(path.join(here, filename), "utf8");
  await workbook.fromCSV(csvText, { sheetName });
}

const notes = workbook.worksheets.add("ReadMe");
notes.getRange("A1:D1").merge();
notes.getRange("A1:D1").values = [["E1/E2 reproducibility audit"]];
notes.getRange("A3:B10").values = [
  ["Item", "Status"],
  ["28-day five-model export", "Complete: 140 rows"],
  ["Table VII gate", "Passed: 25/25 rounded checks"],
  ["NBeatsx_LSTM checkpoint", "Validated pointwise to four decimals"],
  ["28-day E2", "Complete; mechanism supported only in this window"],
  ["365-day export", "Partial: Persistence and NBeatsx_LSTM only"],
  ["365-day E2", "Partial negative result; all three groups lose to Persistence"],
  ["Training/tuning in this audit", "None"],
];
notes.getRange("A12:D15").merge(true);
notes.getRange("A12:A15").values = [
  ["Do not present the 365-day partial workbook sheets as a five-model experiment."],
  ["Exact NBEATSx, LSTM and TCN trained instances must first be recovered and validated against 222.py."],
  ["No test-period tuning, best-seed selection, variant deletion, or post-hoc metric addition was performed."],
  ["See README.md and PROVENANCE_AND_COMPLIANCE.md for the full audit trail."],
];

for (const [_, sheetName] of imports) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange();
  used.format.autofitColumns();
  used.format.autofitRows();
  sheet.freezePanes.freezeRows(1);
  const columnCount = used.columnCount;
  const lastColumn = String.fromCharCode(64 + Math.min(columnCount, 26));
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format.fill = "#1F4E78";
  header.format.font = { bold: true, color: "#FFFFFF" };
  used.format.font = { name: "Arial", size: 10 };
  header.format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
}

notes.getRange("A1:D1").format.fill = "#1F4E78";
notes.getRange("A1:D1").format.font = { name: "Arial", size: 16, bold: true, color: "#FFFFFF" };
notes.getRange("A3:B3").format.fill = "#D9EAF7";
notes.getRange("A3:B3").format.font = { name: "Arial", size: 10, bold: true, color: "#000000" };
notes.getRange("A1:D15").format.wrapText = true;
notes.getRange("A1:D15").format.autofitRows();
notes.getRange("A:A").format.columnWidth = 30;
notes.getRange("B:B").format.columnWidth = 52;
notes.getRange("C:D").format.columnWidth = 18;
notes.showGridLines = false;

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 7000,
  tableMaxRows: 5,
  tableMaxCols: 8,
});
await fs.writeFile(path.join(here, "audit_workbook_inspection.txt"), inspection.ndjson ?? String(inspection), "utf8");

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 5000,
});
await fs.writeFile(path.join(here, "audit_workbook_formula_error_scan.txt"), errors.ndjson ?? String(errors), "utf8");

const preview = await workbook.render({ sheetName: "ReadMe", autoCrop: "all", scale: 1.2, format: "png" });
await fs.writeFile(path.join(here, "audit_workbook_preview.png"), new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(here, "E1_E2_audit.xlsx"));
