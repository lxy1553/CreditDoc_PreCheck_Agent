const fieldLabels = { monthly_income: "月收入", full_name: "姓名", applicant_name: "姓名", id_number: "身份证号" };
const documentLabels = {
  application_form: "申请表", income_proof: "收入证明", identity_document: "身份证明",
  business_license: "营业执照", bank_statement: "银行流水",
};
const numericFields = new Set(["monthly_income", "annual_income", "loan_amount"]);
const jobReasonLabels = {
  queue_timeout: "排队超时",
  hard_deadline: "达到处理时限",
  backlog_limit: "队列已满",
  worker_restart: "服务重启后转人工",
  result_unavailable_after_restart: "服务重启后结果不可恢复",
  worker_payload_unavailable: "任务数据不可用",
  processing_error: "处理异常",
};

export function formatHistoryEntry(job, locale = "zh-CN") {
  const timestamp = Number(job?.created_at);
  const created = Number.isFinite(timestamp) ? new Date(timestamp * 1000).toLocaleString(locale) : "时间未知";
  const status = job?.workbench_status || job?.status || "预审处理中";
  const reason = job?.reason ? ` · ${jobReasonLabels[job.reason] || job.reason}` : "";
  return `${status} · ${created}${reason}`;
}

function readableNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 }).format(number) : String(value ?? "未知");
}

export function formatIssue(issue) {
  const details = issue?.details || {};
  const rows = [];
  if (issue?.category === "field_difference") {
    if (details.field) rows.push(`核对字段：${fieldLabels[details.field] || details.field}`);
    for (const item of details.observations || []) {
      const source = item.source_ref ? `（${item.source_ref}）` : "";
      const value = numericFields.has(details.field) ? readableNumber(item.value) : String(item.value ?? "未知");
      rows.push(`${documentLabels[item.document_type] || item.document_type || "材料"}：${value}${source}`);
    }
    for (const item of details.numeric_differences || []) rows.push(`数值差：${readableNumber(item.difference)}`);
  } else if (issue?.category === "missing_material") {
    const types = (details.accepted_document_types || []).map((type) => documentLabels[type] || type);
    if (types.length) rows.push(`尚未提交：${types.join("、")}`);
  } else if (issue?.category === "low_ocr_confidence") {
    if (details.field) rows.push(`待复核字段：${fieldLabels[details.field] || details.field}`);
    if (Number.isFinite(Number(details.confidence))) rows.push(`识别置信度：${Math.round(Number(details.confidence) * 100)}%`);
    if (details.source_ref) rows.push(`材料位置：${details.source_ref}`);
  }

  for (const item of issue?.rule_evidence || []) {
    const citation = item.citation || {};
    const lines = citation.lines;
    const location = [citation.title, citation.version, citation.section,
      Array.isArray(lines) && lines.length === 2 ? `第 ${lines[0]}-${lines[1]} 行` : null]
      .filter(Boolean).join(" · ");
    if (location) rows.push(`依据：${location}`);
    if (item.evidence) rows.push(String(item.evidence));
  }
  return rows;
}

export async function pollJobUntilTerminal(jobId, fetchJob, {
  onUpdate = () => {}, intervalMs = 350, maxDurationMs = 180_000,
  sleep = (duration) => new Promise((resolve) => setTimeout(resolve, duration)),
} = {}) {
  let elapsed = 0;
  let latest = null;
  while (elapsed < maxDurationMs) {
    await sleep(intervalMs);
    elapsed += intervalMs;
    latest = await fetchJob(jobId);
    onUpdate(latest);
    if (["completed", "manual_review"].includes(latest.status)) {
      return { kind: "terminal", job: latest };
    }
  }
  return { kind: "pending", job: latest };
}

export function presentJob(job) {
  const reason = job.reason ? ` · ${job.reason}` : "";
  if (job.status === "manual_review") {
    return {
      kind: "manual_review", status: (job.workbench_status || "转人工预审") + reason,
      tone: "warn", terminal: true, result: null,
      message: "任务已转人工预审。迟到的 Agent 结果不会覆盖此状态。",
    };
  }
  if (job.status === "completed") {
    return { kind: "completed", status: job.workbench_status || "Agent预审完成", tone: "ok",
      terminal: true, result: job.result ?? null, message: null };
  }
  return { kind: "processing", status: (job.workbench_status || "预审处理中") + reason,
    tone: "", terminal: false, result: null, message: null };
}
