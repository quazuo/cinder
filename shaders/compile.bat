@echo off
for /D %%i in (C:\VulkanSDK\*) do set "SDK_DIR=%%i"
set "IS_ERROR=0"

set graphics_shaders="main" "skybox" "prepass" "sphere-cube" "ss-quad" "ssao"
set rt_shaders="raytrace"

set graphics_exts="vert" "frag"
set rt_exts="rchit" "rgen" "rmiss"

set SPV_FLAGS=-g --target-env=vulkan1.2

(for %%a in (%graphics_shaders%) do (
    (for %%e in (%graphics_exts%) do (
        @echo on
        %SDK_DIR%/Bin/glslc.exe %%a.%%e -o obj/%%a-%%e.spv %SPV_FLAGS%
        @echo off
        if %ERRORLEVEL% NEQ 0 set "IS_ERROR=1"
    ))
))

(for %%a in (%rt_shaders%) do (
    (for %%e in (%rt_exts%) do (
        @echo on
        %SDK_DIR%/Bin/glslc.exe %%a.%%e -o obj/%%a-%%e.spv %SPV_FLAGS%
        @echo off
        if %ERRORLEVEL% NEQ 0 set "IS_ERROR=1"
    ))
))

if %IS_ERROR% NEQ 0 exit 1
