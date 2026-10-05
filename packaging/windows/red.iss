; Inno Setup script for red's Windows installer.
;
; Built by make_zip.ps1 when Inno Setup 6 is installed (winget install
; JRSoftware.InnoSetup), from the same dist\red folder the zip holds:
;     ISCC.exe /DAppVersion=<version> /DStageDir=<dist\red> /DOutDir=<dist> red.iss
;
; Installs per user by default -- no admin, into %LOCALAPPDATA%\Programs\red --
; with the option to install for all users. Start menu entry, optional desktop
; shortcut, an uninstaller in Settings > Apps. Uninstalling removes the program
; only: projects and settings (%USERPROFILE%\red_data, %USERPROFILE%\.config\red)
; are the user's and stay.

#ifndef AppVersion
  #define AppVersion "dev"
#endif
#ifndef StageDir
  #define StageDir "..\..\dist\red"
#endif
#ifndef OutDir
  #define OutDir "..\..\dist"
#endif

[Setup]
; Fixed for every release: it is how a newer installer finds and replaces an
; installed red. Never change it.
AppId={{8FC2728F-DD73-4AE3-97A6-BC5A2377519B}
AppName=Red
AppVersion={#AppVersion}
AppVerName=Red {#AppVersion}
AppPublisher=moments-behavior
AppPublisherURL=https://github.com/moments-behavior/red
AppSupportURL=https://github.com/moments-behavior/red/issues
DefaultDirName={autopf}\red
DefaultGroupName=Red
DisableProgramGroupPage=yes
; Per user unless the user picks "all users" in the dialog.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutDir}
OutputBaseFilename=red-{#AppVersion}-windows-x64-setup
SetupIconFile=red.ico
UninstallDisplayIcon={app}\bin\red.exe
UninstallDisplayName=Red
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
; The whole folder as the zip has it: bin\red.exe and its DLLs, fonts\, the
; default layout. ignoreversion: red's DLLs are replaced as a set on upgrade.
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[InstallDelete]
; An upgrade drops the previous version's DLLs first, so none that the new one
; no longer uses are left behind.
Type: filesandordirs; Name: "{app}\bin"
; Shortcuts from versions that named them "red": Windows matches names without
; case, so overwriting would keep the old lowercase name. Remove, then recreate.
Type: files; Name: "{autoprograms}\red.lnk"
Type: files; Name: "{autodesktop}\red.lnk"

[Icons]
Name: "{autoprograms}\Red"; Filename: "{app}\bin\red.exe"; WorkingDir: "{app}\bin"
Name: "{autodesktop}\Red"; Filename: "{app}\bin\red.exe"; WorkingDir: "{app}\bin"; Tasks: desktopicon

[Run]
Filename: "{app}\bin\red.exe"; Description: "Launch Red"; WorkingDir: "{app}\bin"; Flags: nowait postinstall skipifsilent
