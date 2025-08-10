#!/bin/bash
# HA ML Predictor Version Manager
# Sprint 7 Task 3: CI/CD Pipeline Enhancement & Deployment Automation
#
# This script provides comprehensive version management with semantic versioning,
# changelog generation, and release preparation capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION_FILE="$PROJECT_ROOT/VERSION"
CHANGELOG_FILE="$PROJECT_ROOT/CHANGELOG.md"

# Default values
ACTION=""
VERSION_TYPE="patch"
CUSTOM_VERSION=""
DRY_RUN=false
VERBOSE=false
AUTO_COMMIT=false
SKIP_CHANGELOG=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
HA ML Predictor Version Manager

Usage: $0 ACTION [OPTIONS]

Actions:
    show                       Show current version information
    bump [TYPE]               Bump version by type (major, minor, patch)
    set VERSION               Set specific version
    changelog                 Generate/update changelog
    prepare-release           Prepare release (bump + changelog + commit)
    list-releases             List all releases/tags
    compare VERSION1 VERSION2 Compare two versions

Options:
    -t, --type TYPE           Version bump type (major, minor, patch, prerelease)
    -v, --version VERSION     Custom version to set
    -d, --dry-run            Show what would be done without making changes
    --verbose                Enable verbose output
    --auto-commit            Automatically commit version changes
    --skip-changelog         Skip changelog generation
    -h, --help               Show this help message

Examples:
    $0 show
    $0 bump minor
    $0 set 1.2.3
    $0 prepare-release --type major --auto-commit
    $0 changelog --verbose

EOF
}

# Parse command line arguments
parse_args() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 1
    fi
    
    ACTION="$1"
    shift
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                VERSION_TYPE="$2"
                shift 2
                ;;
            -v|--version)
                CUSTOM_VERSION="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --auto-commit)
                AUTO_COMMIT=true
                shift
                ;;
            --skip-changelog)
                SKIP_CHANGELOG=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                # Handle positional arguments for specific actions
                case $ACTION in
                    set)
                        if [[ -z "$CUSTOM_VERSION" ]]; then
                            CUSTOM_VERSION="$1"
                        fi
                        shift
                        ;;
                    bump)
                        if [[ -z "$VERSION_TYPE" ]] || [[ "$VERSION_TYPE" == "patch" ]]; then
                            VERSION_TYPE="$1"
                        fi
                        shift
                        ;;
                    compare)
                        if [[ -z "$VERSION1" ]]; then
                            VERSION1="$1"
                        elif [[ -z "$VERSION2" ]]; then
                            VERSION2="$1"
                        fi
                        shift
                        ;;
                    *)
                        log_error "Unknown option: $1"
                        show_help
                        exit 1
                        ;;
                esac
                ;;
        esac
    done
}

# Get current version
get_current_version() {
    if [[ -f "$VERSION_FILE" ]]; then
        cat "$VERSION_FILE"
    elif git describe --tags --abbrev=0 2>/dev/null; then
        git describe --tags --abbrev=0 | sed 's/^v//'
    else
        echo "0.0.0"
    fi
}

# Validate version format
validate_version() {
    local version="$1"
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?$ ]]; then
        log_error "Invalid version format: $version"
        log_error "Expected format: MAJOR.MINOR.PATCH[-PRERELEASE]"
        return 1
    fi
    return 0
}

# Calculate next version
calculate_next_version() {
    local current="$1"
    local type="$2"
    
    log_debug "Calculating next version: $current -> $type"
    
    # Parse current version
    if [[ "$current" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-(.+))?$ ]]; then
        local major="${BASH_REMATCH[1]}"
        local minor="${BASH_REMATCH[2]}"
        local patch="${BASH_REMATCH[3]}"
        local prerelease="${BASH_REMATCH[5]}"
    else
        log_error "Cannot parse current version: $current"
        return 1
    fi
    
    log_debug "Parsed: major=$major, minor=$minor, patch=$patch, prerelease=$prerelease"
    
    # Calculate next version based on type
    case "$type" in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            prerelease=""
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            prerelease=""
            ;;
        patch)
            patch=$((patch + 1))
            prerelease=""
            ;;
        prerelease)
            if [[ -n "$prerelease" ]]; then
                # Increment existing prerelease
                if [[ "$prerelease" =~ ^(.+)\.([0-9]+)$ ]]; then
                    local pre_base="${BASH_REMATCH[1]}"
                    local pre_num="${BASH_REMATCH[2]}"
                    prerelease="${pre_base}.$((pre_num + 1))"
                else
                    prerelease="${prerelease}.1"
                fi
            else
                # Create new prerelease
                patch=$((patch + 1))
                prerelease="rc.1"
            fi
            ;;
        *)
            log_error "Invalid version type: $type"
            return 1
            ;;
    esac
    
    # Construct next version
    local next_version="$major.$minor.$patch"
    if [[ -n "$prerelease" ]]; then
        next_version="$next_version-$prerelease"
    fi
    
    echo "$next_version"
}

# Update version in files
update_version_in_files() {
    local version="$1"
    
    log_info "Updating version to $version in project files..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would update version in files"
        return 0
    fi
    
    # Update VERSION file
    echo "$version" > "$VERSION_FILE"
    log_debug "Updated $VERSION_FILE"
    
    # Update version in Python files
    find "$PROJECT_ROOT/src" -name "*.py" -type f -exec grep -l "__version__" {} \; | while read -r file; do
        sed -i "s/__version__ = .*/__version__ = \"$version\"/" "$file"
        log_debug "Updated version in $file"
    done
    
    # Update Docker files if they contain version references
    if [[ -f "$PROJECT_ROOT/docker/Dockerfile" ]]; then
        if grep -q "LABEL version=" "$PROJECT_ROOT/docker/Dockerfile"; then
            sed -i "s/LABEL version=.*/LABEL version=\"$version\"/" "$PROJECT_ROOT/docker/Dockerfile"
            log_debug "Updated version in Dockerfile"
        fi
    fi
    
    # Update docker-compose files
    for compose_file in "$PROJECT_ROOT/docker/docker-compose"*.yml; do
        if [[ -f "$compose_file" ]] && grep -q "image.*:.*" "$compose_file"; then
            sed -i "s/image: ha-ml-predictor:.*/image: ha-ml-predictor:$version/" "$compose_file"
            log_debug "Updated version in $(basename "$compose_file")"
        fi
    done
    
    log_success "Version updated in all project files"
}

# Generate changelog entry
generate_changelog_entry() {
    local version="$1"
    local current_version="$2"
    
    log_info "Generating changelog entry for version $version..."
    
    local release_date
    release_date=$(date +%Y-%m-%d)
    
    # Get commits since last version
    local commits
    if git rev-parse "v$current_version" &>/dev/null; then
        commits=$(git log --pretty=format:"- %s (%h)" "v$current_version..HEAD")
    else
        commits=$(git log --pretty=format:"- %s (%h)" -10)
    fi
    
    # Categorize commits
    local features=""
    local fixes=""
    local improvements=""
    local breaking=""
    local other=""
    
    while IFS= read -r commit; do
        if echo "$commit" | grep -qiE "(feat:|feature:)"; then
            features="$features$commit"$'\n'
        elif echo "$commit" | grep -qiE "(fix:|bug:)"; then
            fixes="$fixes$commit"$'\n'
        elif echo "$commit" | grep -qiE "(improve:|enhancement:|refactor:)"; then
            improvements="$improvements$commit"$'\n'
        elif echo "$commit" | grep -qiE "(breaking|BREAKING)"; then
            breaking="$breaking$commit"$'\n'
        else
            other="$other$commit"$'\n'
        fi
    done <<< "$commits"
    
    # Generate changelog entry
    local changelog_entry="## [v$version] - $release_date
"
    
    if [[ -n "$breaking" ]]; then
        changelog_entry="$changelog_entry
### ⚠️ Breaking Changes
$breaking"
    fi
    
    if [[ -n "$features" ]]; then
        changelog_entry="$changelog_entry
### 🚀 Added
$features"
    fi
    
    if [[ -n "$improvements" ]]; then
        changelog_entry="$changelog_entry
### 🔧 Changed
$improvements"
    fi
    
    if [[ -n "$fixes" ]]; then
        changelog_entry="$changelog_entry
### 🐛 Fixed
$fixes"
    fi
    
    if [[ -n "$other" ]]; then
        changelog_entry="$changelog_entry
### 📝 Other
$other"
    fi
    
    changelog_entry="$changelog_entry
"
    
    echo "$changelog_entry"
}

# Update changelog
update_changelog() {
    local version="$1"
    local current_version="$2"
    
    if [[ "$SKIP_CHANGELOG" == true ]]; then
        log_info "Skipping changelog update"
        return 0
    fi
    
    log_info "Updating changelog..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would update changelog"
        generate_changelog_entry "$version" "$current_version"
        return 0
    fi
    
    local changelog_entry
    changelog_entry=$(generate_changelog_entry "$version" "$current_version")
    
    # Create changelog if it doesn't exist
    if [[ ! -f "$CHANGELOG_FILE" ]]; then
        echo "# Changelog" > "$CHANGELOG_FILE"
        echo "" >> "$CHANGELOG_FILE"
        echo "All notable changes to this project will be documented in this file." >> "$CHANGELOG_FILE"
        echo "" >> "$CHANGELOG_FILE"
        echo "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)," >> "$CHANGELOG_FILE"
        echo "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)." >> "$CHANGELOG_FILE"
        echo "" >> "$CHANGELOG_FILE"
    fi
    
    # Insert new entry at the top
    {
        head -n 6 "$CHANGELOG_FILE"  # Keep header
        echo "$changelog_entry"
        tail -n +7 "$CHANGELOG_FILE"  # Rest of the file
    } > "$CHANGELOG_FILE.tmp"
    
    mv "$CHANGELOG_FILE.tmp" "$CHANGELOG_FILE"
    
    log_success "Changelog updated with v$version entry"
}

# Commit changes
commit_changes() {
    local version="$1"
    
    if [[ "$AUTO_COMMIT" != true ]]; then
        log_info "Auto-commit disabled. Manual commit required."
        return 0
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would commit version changes"
        return 0
    fi
    
    log_info "Committing version changes..."
    
    # Add files to git
    git add "$VERSION_FILE"
    
    # Add Python files with version updates
    find "$PROJECT_ROOT/src" -name "*.py" -type f -exec git add {} \;
    
    # Add Docker files if they were updated
    git add "$PROJECT_ROOT/docker/" 2>/dev/null || true
    
    # Add changelog if it was updated
    if [[ "$SKIP_CHANGELOG" != true ]]; then
        git add "$CHANGELOG_FILE"
    fi
    
    # Commit changes
    git commit -m "chore: bump version to v$version

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Create tag
    git tag -a "v$version" -m "Release v$version"
    
    log_success "Changes committed and tagged as v$version"
    
    # Offer to push
    if [[ -t 1 ]]; then  # Check if running in terminal
        read -p "Push changes and tags to remote? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push origin HEAD
            git push origin "v$version"
            log_success "Changes pushed to remote"
        fi
    fi
}

# Action implementations
action_show() {
    log_info "=== Version Information ==="
    
    local current_version
    current_version=$(get_current_version)
    
    echo "Current Version: $current_version"
    
    if [[ -f "$VERSION_FILE" ]]; then
        echo "Version File: $VERSION_FILE"
    fi
    
    # Show last tag
    if git describe --tags --abbrev=0 2>/dev/null; then
        echo "Last Git Tag: $(git describe --tags --abbrev=0)"
    else
        echo "Last Git Tag: None"
    fi
    
    # Show commits since last tag/version
    local commits_count
    if git describe --tags --abbrev=0 2>/dev/null; then
        commits_count=$(git rev-list "$(git describe --tags --abbrev=0)..HEAD" --count)
        echo "Commits since last tag: $commits_count"
    fi
    
    # Show suggested next versions
    echo ""
    echo "Suggested next versions:"
    echo "  Major: $(calculate_next_version "$current_version" "major")"
    echo "  Minor: $(calculate_next_version "$current_version" "minor")"
    echo "  Patch: $(calculate_next_version "$current_version" "patch")"
    echo "  Prerelease: $(calculate_next_version "$current_version" "prerelease")"
}

action_bump() {
    local current_version
    current_version=$(get_current_version)
    
    local next_version
    next_version=$(calculate_next_version "$current_version" "$VERSION_TYPE")
    
    if ! validate_version "$next_version"; then
        exit 1
    fi
    
    log_info "Bumping version: $current_version → $next_version"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Version bump simulation completed"
        return 0
    fi
    
    update_version_in_files "$next_version"
    update_changelog "$next_version" "$current_version"
    commit_changes "$next_version"
    
    log_success "Version bumped to $next_version"
}

action_set() {
    if [[ -z "$CUSTOM_VERSION" ]]; then
        log_error "No version specified for set action"
        exit 1
    fi
    
    if ! validate_version "$CUSTOM_VERSION"; then
        exit 1
    fi
    
    local current_version
    current_version=$(get_current_version)
    
    log_info "Setting version: $current_version → $CUSTOM_VERSION"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Version set simulation completed"
        return 0
    fi
    
    update_version_in_files "$CUSTOM_VERSION"
    update_changelog "$CUSTOM_VERSION" "$current_version"
    commit_changes "$CUSTOM_VERSION"
    
    log_success "Version set to $CUSTOM_VERSION"
}

action_changelog() {
    local current_version
    current_version=$(get_current_version)
    
    log_info "Generating changelog for current version: $current_version"
    
    local changelog_entry
    changelog_entry=$(generate_changelog_entry "$current_version" "$(get_current_version)")
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "$changelog_entry"
        return 0
    fi
    
    update_changelog "$current_version" "$(get_current_version)"
    
    log_success "Changelog updated"
}

action_prepare_release() {
    log_info "Preparing release..."
    
    local current_version
    current_version=$(get_current_version)
    
    local next_version
    next_version=$(calculate_next_version "$current_version" "$VERSION_TYPE")
    
    log_info "Preparing release: $current_version → $next_version"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Release preparation simulation completed"
        return 0
    fi
    
    update_version_in_files "$next_version"
    update_changelog "$next_version" "$current_version"
    commit_changes "$next_version"
    
    log_success "Release v$next_version prepared and ready!"
    
    echo ""
    echo "Next steps:"
    echo "1. Review the changes: git log -1 --stat"
    echo "2. Push to trigger release workflow: git push origin HEAD && git push origin v$next_version"
    echo "3. Monitor the release pipeline in GitHub Actions"
}

action_list_releases() {
    log_info "=== Release History ==="
    
    if git tag -l | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+' | sort -V; then
        echo ""
        echo "Latest releases:"
        git tag -l --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+' | head -10 | while read -r tag; do
            local date
            date=$(git log -1 --format="%ad" --date=short "$tag" 2>/dev/null || echo "unknown")
            echo "  $tag ($date)"
        done
    else
        echo "No releases found"
    fi
}

# Main execution
main() {
    parse_args "$@"
    
    case "$ACTION" in
        show)
            action_show
            ;;
        bump)
            action_bump
            ;;
        set)
            action_set
            ;;
        changelog)
            action_changelog
            ;;
        prepare-release)
            action_prepare_release
            ;;
        list-releases)
            action_list_releases
            ;;
        *)
            log_error "Unknown action: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi