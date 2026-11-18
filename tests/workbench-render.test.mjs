import test from "node:test";
import assert from "node:assert/strict";
import { formatHistoryEntry, formatIssue, pollJobUntilTerminal, presentJob } from "../src/creditdoc_precheck_agent/static/workbench-render.js";

test("formats a recent job with its terminal state and manual-review reason", () => {
  const label = formatHistoryEntry({
    status: "manual_review", workbench_status: "转人工预审",
    reason: "hard_deadline", created_at: 0,
  });
  assert.match(label, /^转人工预审 · /);
  assert.match(label, /· 达到处理时限$/);
});

test("formats a cross-document mismatch with the observed values and numeric gap", () => {
  const rows = formatIssue({
    category: "field_difference",
    details: {
      field: "monthly_income",
      observations: [
        { document_type: "application_form", source_ref: "申请表.pdf#page=1", value: "12000" },
        { document_type: "income_proof", source_ref: "收入证明.pdf#page=2", value: "10000" },
      ],
      numeric_differences: [{ difference: "2000" }],
    },
  });

  assert.deepEqual(rows, [
    "核对字段：月收入",
    "申请表：12,000（申请表.pdf#page=1）",
    "收入证明：10,000（收入证明.pdf#page=2）",
    "数值差：2,000",
  ]);
});

test("keeps identity numbers unchanged while using the configured field label", () => {
  assert.deepEqual(formatIssue({ category: "field_difference", details: {
    field: "id_number",
    observations: [
      { document_type: "application_form", value: "110105199001011234" },
      { document_type: "identity_document", value: "110105199001011234" },
    ],
  } }), [
    "核对字段：身份证号",
    "申请表：110105199001011234",
    "身份证明：110105199001011234",
  ]);
});

test("formats missing materials, low confidence, and rule citations for manual review", () => {
  assert.deepEqual(formatIssue({ category: "missing_material", details: {
    accepted_document_types: ["income_proof", "identity_document"],
  } }), ["尚未提交：收入证明、身份证明"]);

  assert.deepEqual(formatIssue({ category: "low_ocr_confidence", details: {
    field: "full_name", confidence: 0.62, source_ref: "申请表.pdf#page=1",
  } }), ["待复核字段：姓名", "识别置信度：62%", "材料位置：申请表.pdf#page=1"]);

  assert.deepEqual(formatIssue({ category: "field_difference", rule_evidence: [{
    evidence: "收入证明字段需与申请表核对", citation: {
      title: "贷前材料清单", version: "模拟版 1.0", section: "收入证明", lines: [4, 6],
    },
  }] }), ["依据：贷前材料清单 · 模拟版 1.0 · 收入证明 · 第 4-6 行", "收入证明字段需与申请表核对"]);
});

test("stops polling at manual review so a late Agent result cannot replace that terminal state", async () => {
  const responses = [
    { status: "processing" },
    { status: "manual_review", reason: "hard_deadline" },
    { status: "completed", result: { late: true } },
  ];
  let calls = 0;
  const outcome = await pollJobUntilTerminal("job-1", async () => responses[calls++], {
    intervalMs: 1, maxDurationMs: 10, sleep: async () => {},
  });

  assert.equal(outcome.kind, "terminal");
  assert.equal(outcome.job.status, "manual_review");
  assert.equal(calls, 2);
});

test("returns pending when the client polling window ends before a server terminal state", async () => {
  const outcome = await pollJobUntilTerminal("job-2", async () => ({ status: "processing" }), {
    intervalMs: 1, maxDurationMs: 2, sleep: async () => {},
  });

  assert.equal(outcome.kind, "pending");
  assert.equal(outcome.job.status, "processing");
});

test("presents manual review as terminal and discards any result attached to that state", () => {
  const view = presentJob({
    status: "manual_review", workbench_status: "转人工预审", reason: "hard_deadline",
    result: { late: true },
  });

  assert.deepEqual(view, {
    kind: "manual_review", status: "转人工预审 · hard_deadline", tone: "warn",
    terminal: true, result: null,
    message: "任务已转人工预审。迟到的 Agent 结果不会覆盖此状态。",
  });
});
