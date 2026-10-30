#!/usr/bin/env bash
# Cleanup script to remove deprecated modules and legacy code
# Run this after validating Nix environment works correctly

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  bloodBender Deprecated Code Cleanup Script${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "cleanup/deprecated-modules-and-nix-migration" ]; then
    echo -e "${RED}❌ Error: Not on cleanup branch${NC}"
    echo "Current branch: $CURRENT_BRANCH"
    echo "Expected: cleanup/deprecated-modules-and-nix-migration"
    echo ""
    echo "Run: git checkout cleanup/deprecated-modules-and-nix-migration"
    exit 1
fi

echo -e "${GREEN}✅ On correct branch: $CURRENT_BRANCH${NC}"
echo ""

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted changes${NC}"
    echo -e "${YELLOW}   Consider committing or stashing them first${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

# Dry run mode
DRY_RUN=false
if [ "${1:-}" == "--dry-run" ]; then
    DRY_RUN=true
    echo -e "${YELLOW}🔍 DRY RUN MODE - No files will be deleted${NC}"
    echo ""
fi

# Function to remove directory
remove_dir() {
    local dir=$1
    local reason=$2
    
    if [ -d "$dir" ]; then
        local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo -e "${YELLOW}📁 $dir${NC} ($size)"
        echo -e "   Reason: $reason"
        
        if [ "$DRY_RUN" = false ]; then
            git rm -rf "$dir" 2>/dev/null || rm -rf "$dir"
            echo -e "   ${GREEN}✅ Removed${NC}"
        else
            echo -e "   ${YELLOW}[DRY RUN] Would remove${NC}"
        fi
    else
        echo -e "${YELLOW}📁 $dir${NC}"
        echo -e "   ${GREEN}Already removed or doesn't exist${NC}"
    fi
    echo ""
}

# Function to remove files matching pattern
remove_pattern() {
    local pattern=$1
    local reason=$2
    
    echo -e "${YELLOW}🗑️  Removing files matching: $pattern${NC}"
    echo -e "   Reason: $reason"
    
    local count=$(find . -maxdepth 1 -name "$pattern" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo -e "   Found $count files"
        if [ "$DRY_RUN" = false ]; then
            find . -maxdepth 1 -name "$pattern" -exec git rm -f {} \; 2>/dev/null || \
            find . -maxdepth 1 -name "$pattern" -delete
            echo -e "   ${GREEN}✅ Removed${NC}"
        else
            echo -e "   ${YELLOW}[DRY RUN] Would remove:${NC}"
            find . -maxdepth 1 -name "$pattern" -exec basename {} \; | head -5
            if [ "$count" -gt 5 ]; then
                echo -e "   ... and $(($count - 5)) more"
            fi
        fi
    else
        echo -e "   ${GREEN}No files found${NC}"
    fi
    echo ""
}

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 1: Deprecated Directories"
echo "═══════════════════════════════════════════════════════════════"
echo ""

remove_dir "sweetBloodDeprecated" "Replaced by bloodBank v2.0 architecture"
remove_dir "bloodBath-env.bak" "Backup of old virtual environment (~500MB)"
remove_dir "training_data_legacy" "Pre-v2.0 training data format"
remove_dir "test_fixed_v2" "Old test output directory"
remove_dir "test_logs" "Historical test logs"
remove_dir "test_monthly_validation" "Old validation test artifacts"
remove_dir "test_results" "Historical test results"
remove_dir "unified_lstm_training" "Replaced by bloodTwin module"

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 2: Old Log Files"
echo "═══════════════════════════════════════════════════════════════"
echo ""

remove_pattern "bloodbank_*.log" "Historical operation logs"
remove_pattern "csv_repair_*.log" "CSV repair operation logs"
remove_pattern "csv_repair_summary_*.json" "CSV repair summaries"
remove_pattern "test_*.log" "Test execution logs"
remove_pattern "*.log" "Other miscellaneous logs"

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 3: Root-Level Test Scripts"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo -e "${YELLOW}⚠️  Found root-level test scripts:${NC}"
TEST_FILES=$(find . -maxdepth 1 -name "test_*.py" 2>/dev/null | wc -l)
if [ "$TEST_FILES" -gt 0 ]; then
    echo -e "   $TEST_FILES test_*.py files"
    echo ""
    echo -e "${YELLOW}   These should be reviewed and either:${NC}"
    echo "   1. Moved to bloodBath/test_scripts/"
    echo "   2. Removed if obsolete"
    echo ""
    echo "   Files found:"
    find . -maxdepth 1 -name "test_*.py" -exec basename {} \; | sort
    echo ""
    
    if [ "$DRY_RUN" = false ]; then
        read -p "Remove all root-level test_*.py files? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            find . -maxdepth 1 -name "test_*.py" -exec git rm -f {} \; 2>/dev/null || \
            find . -maxdepth 1 -name "test_*.py" -delete
            echo -e "   ${GREEN}✅ Removed test files${NC}"
        else
            echo -e "   ${YELLOW}⏭️  Skipped - review manually${NC}"
        fi
    else
        echo -e "   ${YELLOW}[DRY RUN] Would prompt for removal${NC}"
    fi
else
    echo -e "   ${GREEN}No root-level test files found${NC}"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 4: Deprecated Python Stubs"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo -e "${YELLOW}📝 Code cleanup required:${NC}"
echo ""
echo "The following files contain deprecated compatibility stubs:"
echo ""
echo "1. bloodBath/cli/main.py (lines 38-56)"
echo "   - add_sweetblood_args()"
echo "   - handle_sweetblood_commands()"
echo "   - SweetBloodIntegration class"
echo ""
echo "2. bloodBath/utils/structure_utils.py"
echo "   - setup_sweetblood_environment()"
echo ""
echo "3. bloodBath/data/processors.py (line 921+)"
echo "   - Legacy DataProcessor wrapper"
echo ""
echo -e "${YELLOW}⚠️  Manual review recommended before removal${NC}"
echo "   Run: grep -r 'sweetblood\|SweetBlood' bloodBath/ --include='*.py'"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}This was a DRY RUN - no files were actually removed${NC}"
    echo ""
    echo "To perform cleanup, run:"
    echo "  ./cleanup_deprecated.sh"
else
    echo -e "${GREEN}✅ Cleanup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review changes: git status"
    echo "  2. Test environment: nix develop"
    echo "  3. Run tests: python -m pytest bloodBath/test_scripts/"
    echo "  4. Commit changes: git commit -m 'Remove deprecated code'"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
