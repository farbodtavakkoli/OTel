{{/*
Names. All resources are scoped by Release.Name so two instances of the
chart can live side-by-side in the same namespace.
*/}}
{{- define "scalarlm.fullname" -}}
{{- printf "%s" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.vllmname" -}}
{{- printf "%s-vllm" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.megatronname" -}}
{{- printf "%s-megatron" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.cloudflaredFullName" -}}
{{- printf "%s-cloudflared" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.cloudflaredSecretName" -}}
{{- if .Values.cloudflared.existingSecret -}}
{{- .Values.cloudflared.existingSecret -}}
{{- else -}}
{{- printf "%s-cloudflared" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/*
Labels. `selectorLabels` is stable across upgrades and used in Deployment
selectors; `labels` is the richer set suitable for metadata.
*/}}
{{- define "scalarlm.commonLabels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "scalarlm.labels" -}}
app.kubernetes.io/name: {{ include "scalarlm.fullname" . }}
app.kubernetes.io/component: api
{{ include "scalarlm.commonLabels" . }}
{{- end -}}
{{- define "scalarlm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "scalarlm.vllmlabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.vllmname" . }}
app.kubernetes.io/component: vllm
{{ include "scalarlm.commonLabels" . }}
{{- end -}}
{{- define "scalarlm.vllmSelectorLabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.vllmname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "scalarlm.megatronlabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.megatronname" . }}
app.kubernetes.io/component: megatron
{{ include "scalarlm.commonLabels" . }}
{{- end -}}
{{- define "scalarlm.megatronSelectorLabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.megatronname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "scalarlm.cloudflaredLabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.cloudflaredFullName" . }}
app.kubernetes.io/component: cloudflared
{{ include "scalarlm.commonLabels" . }}
{{- end -}}
{{- define "scalarlm.cloudflaredSelectorLabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.cloudflaredFullName" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Render a cray-config.yaml body for a given server_list. Accepts a dict:
  { root: ., serverList: "api|vllm|megatron", extra: <map> }
The map in `extra` is merged over `.Values.extraConfig` last.
*/}}
{{- define "scalarlm.crayConfig" -}}
{{- $root := .root -}}
{{- $lines := list -}}
{{- $lines = append $lines (printf "model: %s" $root.Values.model) -}}
{{- $lines = append $lines (printf "server_list: %s" .serverList) -}}
{{- $lines = append $lines (printf "max_train_time: %v" $root.Values.max_train_time) -}}
{{- if or (eq .serverList "api") (eq .serverList "vllm") -}}
{{- $lines = append $lines (printf "gpu_memory_utilization: %v" $root.Values.gpu_memory_utilization) -}}
{{- end -}}
{{- if eq .serverList "api" -}}
{{- $lines = append $lines (printf "vllm_api_url: \"http://%s:%v\"" (include "scalarlm.vllmname" $root) $root.Values.service.vllm_port) -}}
{{- end -}}
{{- if eq .serverList "vllm" -}}
{{- $lines = append $lines (printf "api_url: \"http://%s:%v\"" (include "scalarlm.fullname" $root) $root.Values.service.api_port) -}}
{{- end -}}
{{- range $k, $v := $root.Values.extraConfig -}}
{{- if and (kindIs "string" $v) (eq $v "") -}}
{{/* Preserve explicit empty-string opt-outs such as reasoning_parser: "".
     An unquoted empty YAML value would reload as null and re-enable
     model-name detection. */}}
{{- $lines = append $lines (printf "%s: \"\"" $k) -}}
{{- else -}}
{{- $lines = append $lines (printf "%s: %v" $k $v) -}}
{{- end -}}
{{- end -}}
{{- range $k, $v := .extra -}}
{{- $lines = append $lines (printf "%s: %v" $k $v) -}}
{{- end -}}
{{- join "\n" $lines -}}
{{- end -}}

{{/*
Node affinity block from a list of preferred hostnames. Renders nothing
when the list is empty so the template is a no-op by default.
*/}}
{{- define "scalarlm.nodeAffinity" -}}
{{- if . -}}
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values:
{{- range . }}
                - {{ . | quote }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
Common DNS config — every pod needs to resolve the megatron headless
service for slurmd peer discovery.
*/}}
{{- define "scalarlm.dnsConfig" -}}
dnsConfig:
  searches:
    - {{ include "scalarlm.megatronname" . }}-headless.{{ .Release.Namespace }}.svc.cluster.local
    - "{{ .Release.Namespace }}.svc.cluster.local"
    - "svc.cluster.local"
    - "cluster.local"
  options:
    - name: ndots
      value: "1"
{{- end -}}

{{/*
Render an env list from a map. Usage: include "scalarlm.envFromMap" .map
Keys are converted to env-var names verbatim; values are stringified.
*/}}
{{- define "scalarlm.envFromMap" -}}
{{- range $k, $v := . }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
{{- end -}}
