# Build and Package Script for ClassifierOrganizer


# Clean previous builds
Write-Host "`nCleaning previous builds..." -ForegroundColor Cyan
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "ClassifierOrganizer.zip") { Remove-Item -Force "ClassifierOrganizer.zip" }

# Build the executable
Write-Host "`nBuilding executable with cx_Freeze..." -ForegroundColor Cyan
.\.venv\Scripts\python setup.py build

# Check if build was successful
if (Test-Path "build") {
    Write-Host "`nBuild successful!" -ForegroundColor Green
    
    # Find the build directory (cx_Freeze creates a subdirectory like exe.win-amd64-3.11)
    $buildDir = Get-ChildItem -Path "build" -Directory | Select-Object -First 1
    
    if ($buildDir) {
        Write-Host "`nBuild directory: $($buildDir.FullName)" -ForegroundColor Yellow
        
        # Create zip file
        Write-Host "`nCreating zip archive..." -ForegroundColor Cyan
        $zipName = "ClassifierOrganizer_v2.0_Win.zip"
        Compress-Archive -Path "$($buildDir.FullName)\*" -DestinationPath $zipName -Force
        
        Write-Host "`nPackaging complete!" -ForegroundColor Green
        Write-Host "Executable location: $($buildDir.FullName)\ClassifierOrganizer.exe" -ForegroundColor Yellow
        Write-Host "Zip file created: $zipName" -ForegroundColor Yellow
        Write-Host "`nYou can distribute the zip file to users." -ForegroundColor Cyan
    } else {
        Write-Host "Error: Could not find build directory" -ForegroundColor Red
    }
} else {
    Write-Host "`nBuild failed!" -ForegroundColor Red
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
}
