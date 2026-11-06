import logging
import os

logger = logging.getLogger(__name__)


def init_tracing(app, service_name="scalarlm-api"):
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource

        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True
        )

        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)

        logger.info(f"OpenTelemetry tracing initialized for {service_name}, exporting to {otlp_endpoint}")
        return trace.get_tracer(__name__)

    except ImportError as e:
        logger.warning(f"OpenTelemetry not available: {e}. Tracing disabled.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}. Tracing disabled.")
        return None
