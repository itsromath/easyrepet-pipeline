const runningStages = new Set([
    "audio_prepare",
    "whisper",
    "report_prepare",
    "import",
    "llm_4b",
    "llm_9b",
    "final_report",
]);

function stagePulseClass(data) {
    if (data.error || data.stage === "error") {
        return "stage-loader stage-error";
    }

    if (data.stage === "idle" && data.readiness && data.readiness.ready === false) {
        return "stage-loader stage-error";
    }

    const stage = data.stage || (data.running ? "report_prepare" : "idle");
    return `stage-loader stage-${stage}`;
}

function stageLabelClass(data) {
    if (data.error || data.stage === "error") {
        return "stage-label status-error";
    }

    if (data.stage === "done") {
        return "stage-label status-ready";
    }

    if (data.stage === "idle" && data.readiness) {
        return data.readiness.ready ? "stage-label status-ready" : "stage-label status-error";
    }

    if (data.running || runningStages.has(data.stage)) {
        return "stage-label status-running";
    }

    return "stage-label status-idle";
}

function updateWorkflowSteps(data) {
    const steps = Array.isArray(data.workflow_steps) ? data.workflow_steps : [];
    const byId = new Map(steps.map((step) => [step.id, step]));

    document.querySelectorAll(".workflow-step").forEach((element) => {
        const step = byId.get(element.dataset.stepId);
        const state = step ? step.state : "pending";
        element.className = `workflow-step is-${state}`;
    });
}

function updateReadiness(data) {
    const readiness = data.readiness;
    if (!readiness || data.running || data.stage !== "idle") {
        return;
    }

    const statusText = document.getElementById("statusText");
    const stageDetail = document.getElementById("stageDetail");
    const errorText = document.getElementById("errorText");
    if (statusText) {
        statusText.textContent = readiness.label || statusText.textContent;
    }
    if (stageDetail) {
        stageDetail.textContent = readiness.ready
            ? "Можно запускать обработку."
            : (readiness.issues || []).join("; ");
    }
    if (errorText && readiness.ready) {
        errorText.textContent = "";
    }
}

function updateRunControls(data) {
    const running = Boolean(data.running);
    const processAfter = document.querySelector("[data-process-after]");
    document.querySelectorAll("[data-requires-transcribe]").forEach((button) => {
        button.disabled = running || data.can_transcribe === false;
    });
    document.querySelectorAll("[data-requires-analyze]").forEach((button) => {
        button.disabled = running || data.can_analyze === false;
    });
    document.querySelectorAll("[data-audio-submit]").forEach((button) => {
        const wantsReport = !processAfter || processAfter.checked;
        button.disabled = running || data.can_transcribe === false || (wantsReport && data.can_analyze === false);
    });
}

async function refreshStatus() {
    const statusText = document.getElementById("statusText");
    if (!statusText) {
        return;
    }

    try {
        const response = await fetch("/api/status");
        const data = await response.json();

        const stagePulse = document.getElementById("stagePulse");
        const stageLabel = document.getElementById("stageLabel");
        const stageDetail = document.getElementById("stageDetail");

        if (stagePulse) {
            stagePulse.className = stagePulseClass(data);
        }

        if (stageLabel) {
            stageLabel.textContent = data.stage_label || "Статус";
            stageLabel.className = stageLabelClass(data);
        }

        statusText.textContent = data.status || "-";

        if (stageDetail) {
            stageDetail.textContent = data.stage_detail || "";
        }

        updateWorkflowSteps(data);
        updateReadiness(data);
        updateRunControls(data);

        document.getElementById("targetText").textContent = data.target || "-";
        document.getElementById("startedText").textContent = data.started_display || data.started_at || "-";
        document.getElementById("finishedText").textContent = data.finished_display || data.finished_at || "-";
        const elapsedText = document.getElementById("elapsedText");
        if (elapsedText) {
            elapsedText.textContent = data.elapsed_display || "-";
        }
        document.getElementById("errorText").textContent = data.error || "";

        if (!data.running && window.__wasRunning) {
            window.__wasRunning = false;
            setTimeout(() => window.location.reload(), 800);
        }

        if (data.running) {
            window.__wasRunning = true;
        }
    } catch (error) {
        console.error("Status refresh error:", error);
    }
}

async function refreshLogs() {
    const logContent = document.getElementById("logContent");
    if (!logContent) {
        return;
    }

    try {
        const response = await fetch("/api/logs");
        const data = await response.json();
        logContent.textContent = data.content || "";
    } catch (error) {
        console.error("Logs refresh error:", error);
    }
}

function makeTranscriptSuggestion(sourceName) {
    const stopWords = new Set([
        "audio",
        "video",
        "videoplayback",
        "recording",
        "record",
        "lesson",
        "copy",
        "копия",
        "запись",
        "занятия",
        "урок",
        "аудио",
        "видео",
    ]);
    const stem = String(sourceName || "")
        .split(/[\\/]/)
        .pop()
        .replace(/\.[^.]+$/, "")
        .replace(/^\d{8}[_-]\d{6}[_-]?/, "");
    const words = stem.match(/[A-Za-zА-Яа-яЁё0-9]+/g) || [];
    const selected = [];
    for (const word of words) {
        if (stopWords.has(word.toLowerCase())) {
            continue;
        }
        selected.push(word);
        if (selected.length >= 4) {
            break;
        }
    }
    return (selected.length ? selected : words.slice(0, 4)).join("_");
}

function setTranscriptSuggestion(input, suggestion) {
    if (!input || !suggestion) {
        return;
    }
    input.dataset.suggestion = suggestion;
    input.placeholder = `Tab: ${suggestion}`;
}

function selectedSourceName(select) {
    const option = select.options[select.selectedIndex];
    if (!option || !option.value) {
        return "";
    }
    return option.dataset.sourceName || option.textContent.split("·")[1] || option.value;
}

function setupTranscriptSuggestions() {
    document.querySelectorAll("[data-transcript-source]").forEach((source) => {
        const target = document.getElementById(source.dataset.targetOutput || "");
        if (!target) {
            return;
        }

        const refreshSuggestion = () => {
            const sourceName = source.type === "file"
                ? (source.files && source.files[0] ? source.files[0].name : "")
                : selectedSourceName(source);
            setTranscriptSuggestion(target, makeTranscriptSuggestion(sourceName));
        };

        source.addEventListener("change", refreshSuggestion);
        refreshSuggestion();
    });

    document.querySelectorAll("[data-suggestion-target]").forEach((input) => {
        input.addEventListener("keydown", (event) => {
            if (event.key !== "Tab" || input.value.trim()) {
                return;
            }
            const suggestion = input.dataset.suggestion;
            if (!suggestion) {
                return;
            }
            event.preventDefault();
            input.value = suggestion;
        });
    });
}

const processAfterCheckbox = document.querySelector("[data-process-after]");
if (processAfterCheckbox) {
    processAfterCheckbox.addEventListener("change", refreshStatus);
}

if (document.getElementById("statusText")) {
    setInterval(refreshStatus, 2000);
    refreshStatus();
}

if (document.getElementById("logContent")) {
    const refreshLogsButton = document.getElementById("refreshLogsButton");
    if (refreshLogsButton) {
        refreshLogsButton.addEventListener("click", refreshLogs);
    }
}

setupTranscriptSuggestions();

document.querySelectorAll("[data-student-picker]").forEach((select) => {
    const nameField = document.querySelector(
        `[data-new-student-name="${select.dataset.studentPicker}"]`
    );
    if (!nameField) {
        return;
    }
    const input = nameField.querySelector("input");
    const refreshStudentPicker = () => {
        const isNew = select.value === "__new__";
        nameField.hidden = !isNew;
        if (input) {
            input.required = isNew;
            if (isNew) {
                input.focus();
            }
        }
    };
    select.addEventListener("change", refreshStudentPicker);
    refreshStudentPicker();
});

document.querySelectorAll("[data-open-student-dialog]").forEach((button) => {
    const dialog = document.getElementById(button.dataset.openStudentDialog);
    if (!dialog) {
        return;
    }
    button.addEventListener("click", () => dialog.showModal());
});

document.querySelectorAll(".student-dialog").forEach((dialog) => {
    dialog.querySelectorAll("[data-close-student-dialog]").forEach((button) => {
        button.addEventListener("click", () => dialog.close());
    });
    dialog.addEventListener("click", (event) => {
        if (event.target === dialog) {
            dialog.close();
        }
    });
});
