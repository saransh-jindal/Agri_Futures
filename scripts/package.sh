#!/bin/bash

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
PROJECT_NAME="ml-pipeline"

# Create project structure
mkdir -p "$TEMP_DIR/$PROJECT_NAME"/{src/modeling,config,scripts,tests,.github/{workflows,ISSUE_TEMPLATE}}

# Copy all project files
cp -r src/modeling/* "$TEMP_DIR/$PROJECT_NAME/src/modeling/"
cp -r config/* "$TEMP_DIR/$PROJECT_NAME/config/"
cp -r scripts/* "$TEMP_DIR/$PROJECT_NAME/scripts/"
cp -r tests/* "$TEMP_DIR/$PROJECT_NAME/tests/"
cp -r .github/workflows/* "$TEMP_DIR/$PROJECT_NAME/.github/workflows/"
cp -r .github/ISSUE_TEMPLATE/* "$TEMP_DIR/$PROJECT_NAME/.github/ISSUE_TEMPLATE/"

# Copy root level files
cp README.md "$TEMP_DIR/$PROJECT_NAME/"
cp requirements.txt "$TEMP_DIR/$PROJECT_NAME/"
cp LICENSE "$TEMP_DIR/$PROJECT_NAME/"
cp .gitignore "$TEMP_DIR/$PROJECT_NAME/"

# Create zip file
cd "$TEMP_DIR"
zip -r "$PROJECT_NAME.zip" "$PROJECT_NAME"

# Move zip to current directory
mv "$PROJECT_NAME.zip" "$OLDPWD"

# Cleanup
cd "$OLDPWD"
rm -rf "$TEMP_DIR"

echo "Project packaged as $PROJECT_NAME.zip"
echo "You can now upload this zip file to GitHub" 
