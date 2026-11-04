{{/*
Chart name, truncated to the 63 character DNS label limit.
*/}}
{{- define "llm-serve.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified resource name.
*/}}
{{- define "llm-serve.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart label value.
*/}}
{{- define "llm-serve.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Selector labels, stable across upgrades.
*/}}
{{- define "llm-serve.selectorLabels" -}}
app.kubernetes.io/name: {{ include "llm-serve.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "llm-serve.labels" -}}
helm.sh/chart: {{ include "llm-serve.chart" . }}
{{ include "llm-serve.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
llm-serve/backend: {{ .Values.backend }}
{{- end }}

{{/*
Validate backend and model settings, failing with an explicit message.
*/}}
{{- define "llm-serve.validate" -}}
{{- if not (has .Values.backend (list "llamacpp" "sglang")) }}
{{- fail (printf "values.backend must be \"llamacpp\" or \"sglang\", got %q" (toString .Values.backend)) }}
{{- end }}
{{- if eq .Values.backend "llamacpp" }}
{{- if not .Values.model.repo }}
{{- fail "model.repo is required for the llamacpp backend (set it with --set model.repo=<owner>/<name>)" }}
{{- end }}
{{- if not .Values.model.file }}
{{- fail "model.file is required for the llamacpp backend (set it with --set model.file=<file>.gguf)" }}
{{- end }}
{{- if .Values.model.sha256 }}
{{- if not (regexMatch "^[0-9a-fA-F]{64}$" (toString .Values.model.sha256)) }}
{{- fail "model.sha256 must be exactly 64 hexadecimal characters" }}
{{- end }}
{{- else if not .Values.model.allowUnverified }}
{{- fail "model.sha256 is empty: set model.sha256 to the SHA-256 of the GGUF file, or set model.allowUnverified=true to skip verification explicitly" }}
{{- end }}
{{- end }}
{{- if eq .Values.backend "sglang" }}
{{- if and (not .Values.model.repo) (not .Values.model.path) }}
{{- fail "either model.repo or model.path is required for the sglang backend" }}
{{- end }}
{{- if .Values.model.sha256 }}
{{- fail "model.sha256 is supported only by the llamacpp backend; the sglang backend downloads the snapshot itself" }}
{{- end }}
{{- if not .Values.model.allowUnverified }}
{{- fail "the sglang backend cannot verify a SHA-256 checksum; set model.allowUnverified=true to accept an unverified model" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Container port of the active backend.
*/}}
{{- define "llm-serve.containerPort" -}}
{{- if eq .Values.backend "sglang" }}{{ .Values.sglang.port }}{{ else }}{{ .Values.llamacpp.port }}{{ end }}
{{- end }}

{{/*
Queue-depth metric exposed by the active backend.
*/}}
{{- define "llm-serve.queueMetric" -}}
{{- if eq .Values.backend "sglang" }}sglang:num_queue_reqs{{ else }}llamacpp:requests_deferred{{ end }}
{{- end }}

{{/*
Container resources. The sglang backend receives a GPU limit unless one is set.
*/}}
{{- define "llm-serve.resources" -}}
{{- $res := deepCopy (default (dict) .Values.resources) }}
{{- if eq .Values.backend "sglang" }}
{{- $limits := default (dict) $res.limits }}
{{- if not (hasKey $limits "nvidia.com/gpu") }}
{{- $_ := set $limits "nvidia.com/gpu" 1 }}
{{- end }}
{{- $_ := set $res "limits" $limits }}
{{- end }}
{{- toYaml $res }}
{{- end }}

{{/*
Name of the persistent volume claim holding model files.
*/}}
{{- define "llm-serve.claimName" -}}
{{- default (printf "%s-model" (include "llm-serve.fullname" .)) .Values.model.persistence.existingClaim }}
{{- end }}
