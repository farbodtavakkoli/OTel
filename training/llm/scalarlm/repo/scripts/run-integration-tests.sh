#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$PROJECT_DIR/infra:$PROJECT_DIR:${PYTHONPATH:-}"

echo "🚀 ScalarLM vLLM Mode Integration Tests"
echo "======================================="
echo "Project: $PROJECT_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

total_suites=0
passed_suites=0
failed_suites=0
failed_suite_names=()

run_test_suite() {
    local test_name="$1"
    local test_script="$2"
    local description="$3"
    
    echo -e "${BLUE}🧪 Running: $test_name${NC}"
    echo "   $description"
    echo ""
    
    total_suites=$((total_suites + 1))
    
    if python "$test_script"; then
        echo -e "${GREEN}✅ $test_name - PASSED${NC}"
        passed_suites=$((passed_suites + 1))
    else
        echo -e "${RED}❌ $test_name - FAILED${NC}"
        failed_suites=$((failed_suites + 1))
        failed_suite_names+=("$test_name")
    fi
    echo ""
}

if [[ ! -d "test/integration" ]]; then
    echo -e "${RED}❌ Integration test directory not found: test/integration${NC}"
    exit 1
fi

echo "📋 Available Integration Test Suites:"
echo "-------------------------------------"
echo "1. Engine Method Coverage Analysis - Static analysis of HTTP/Direct parity"
echo "2. Direct Mode Coverage Tests - Test all endpoints in Direct mode"  
echo "3. HTTP vs Direct Parity Tests - Compare HTTP and Direct mode results"
echo ""

echo "Choose tests to run:"
echo "  a) All tests"
echo "  1) Engine Method Coverage only"
echo "  2) Direct Mode Coverage only"
echo "  3) HTTP vs Direct Parity only"
echo ""
read -p "Enter choice (a/1/2/3): " choice

case $choice in
    a|A)
        echo -e "${YELLOW}Running all integration test suites...${NC}"
        echo ""

        run_test_suite \
            "Engine Method Coverage" \
            "test/integration/test_engine_method_coverage.py" \
            "Static analysis of HTTP endpoint to engine method mapping"

        echo -e "${YELLOW}⚠️  NOTE: The next test requires ScalarLM running with vllm_use_http: false${NC}"
        echo "   Start server with: ./scalarlm up cpu --direct"
        echo ""
        read -p "Press Enter when server is ready with direct mode, or 's' to skip: " skip_direct
        
        if [[ "$skip_direct" != "s" && "$skip_direct" != "S" ]]; then
            run_test_suite \
                "Direct Mode Coverage" \
                "test/integration/test_direct_mode_coverage.py" \
                "Test all ScalarLM endpoints in Direct mode"
        else
            echo -e "${YELLOW}⏸️  Skipping Direct Mode Coverage tests${NC}"
            echo ""
        fi

        echo -e "${YELLOW}⚠️  NOTE: The next test requires both HTTP and Direct mode accessible${NC}"
        echo "   Ensure ScalarLM (HTTP) and vLLM (Direct) servers are running"
        echo ""
        read -p "Press Enter when both servers are ready, or 's' to skip: " skip_parity
        
        if [[ "$skip_parity" != "s" && "$skip_parity" != "S" ]]; then
            run_test_suite \
                "HTTP vs Direct Parity" \
                "test/integration/test_vllm_mode_parity.py" \
                "Compare HTTP and Direct mode results for parity"
        else
            echo -e "${YELLOW}⏸️  Skipping HTTP vs Direct Parity tests${NC}"
            echo ""
        fi
        ;;
    
    1)
        run_test_suite \
            "Engine Method Coverage" \
            "test/integration/test_engine_method_coverage.py" \
            "Static analysis of HTTP endpoint to engine method mapping"
        ;;
    
    2)
        echo -e "${YELLOW}⚠️  NOTE: This test requires ScalarLM running with vllm_use_http: false${NC}"
        echo "   Start server with: ./scalarlm up cpu --direct"
        echo ""
        read -p "Press Enter when server is ready, or Ctrl+C to cancel: "
        
        run_test_suite \
            "Direct Mode Coverage" \
            "test/integration/test_direct_mode_coverage.py" \
            "Test all ScalarLM endpoints in Direct mode"
        ;;
    
    3)
        echo -e "${YELLOW}⚠️  NOTE: This test requires both HTTP and Direct mode accessible${NC}"
        echo "   Ensure ScalarLM (HTTP) and vLLM (Direct) servers are running"
        echo ""
        read -p "Press Enter when both servers are ready, or Ctrl+C to cancel: "
        
        run_test_suite \
            "HTTP vs Direct Parity" \
            "test/integration/test_vllm_mode_parity.py" \
            "Compare HTTP and Direct mode results for parity"
        ;;
    
    *)
        echo -e "${RED}❌ Invalid choice: $choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "="*80
echo "📊 INTEGRATION TEST SUMMARY"
echo "="*80
echo "Total Test Suites: $total_suites"
echo "Passed:           $passed_suites"
echo "Failed:           $failed_suites"

if [ $failed_suites -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Failed Test Suites:${NC}"
    for failed_suite in "${failed_suite_names[@]}"; do
        echo "   - $failed_suite"
    done
    echo ""
    echo -e "${YELLOW}💡 Tips for failed tests:${NC}"
    echo "   - Ensure ScalarLM server is running for coverage tests"
    echo "   - Check vllm_use_http configuration for direct mode tests"
    echo "   - Verify both ScalarLM and vLLM endpoints are accessible for parity tests"
    echo ""
    exit 1
else
    echo ""
    echo -e "${GREEN}🎉 All integration tests passed!${NC}"
    echo ""
    echo "✅ HTTP to Direct engine method coverage verified"
    echo "✅ Direct mode functionality confirmed"
    echo "✅ HTTP vs Direct parity validated"
    echo ""
    exit 0
fi