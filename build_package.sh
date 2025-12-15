#!/bin/bash
# Build and Package Script for ClassifierOrganizer (Linux)

set -e  # Exit on error

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Clean previous builds
echo -e "${CYAN}\nCleaning previous builds...${NC}"
rm -rf build
rm -f ClassifierOrganizer_v2.0_Linux.tar.gz

# Build the executable
echo -e "${CYAN}\nBuilding executable with cx_Freeze...${NC}"
./.venv/bin/python setup.py build

# Check if build was successful
if [ -d "build" ]; then
    echo -e "${GREEN}\nBuild successful!${NC}"
    
    # Find the build directory (cx_Freeze creates a subdirectory like exe.linux-x86_64-3.11)
    BUILD_DIR=$(find build -maxdepth 1 -type d -name "exe.*" | head -n 1)
    
    if [ -n "$BUILD_DIR" ]; then
        echo -e "${YELLOW}\nBuild directory: $BUILD_DIR${NC}"
        
        # Create tar.gz archive
        echo -e "${CYAN}\nCreating tar.gz archive...${NC}"
        TAR_NAME="ClassifierOrganizer_v2.0_Linux.tar.gz"
        tar -czf "$TAR_NAME" -C "$BUILD_DIR" .
        
        echo -e "${GREEN}\nPackaging complete!${NC}"
        echo -e "${YELLOW}Executable location: $BUILD_DIR/ClassifierOrganizer${NC}"
        echo -e "${YELLOW}Archive created: $TAR_NAME${NC}"
        echo -e "${CYAN}\nYou can distribute the tar.gz file to users.${NC}"
        echo -e "${CYAN}Users can extract with: tar -xzf $TAR_NAME${NC}"
    else
        echo -e "${RED}Error: Could not find build directory${NC}"
        exit 1
    fi
else
    echo -e "${RED}\nBuild failed!${NC}"
    echo -e "${YELLOW}Check the error messages above for details.${NC}"
    exit 1
fi
