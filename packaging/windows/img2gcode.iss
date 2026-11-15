; Inno Setup script for the Windows installer.
; Build after `cargo build --release`:  iscc /DVersion=0.2.0 packaging\windows\img2gcode.iss

#ifndef Version
  #define Version "0.0.0"
#endif
#define Root "..\.."

[Setup]
AppId={{6A3C2F4E-9B1D-4E7A-8C55-2D1F0B7E4A93}
AppName=img2gcode
AppVersion={#Version}
AppVerName=img2gcode {#Version}
AppPublisher=RowanRama
AppPublisherURL=https://github.com/RowanRama/img2gcode
AppSupportURL=https://github.com/RowanRama/img2gcode/issues
DefaultDirName={autopf}\img2gcode
DefaultGroupName=img2gcode
DisableProgramGroupPage=yes
; Installs for the current user without admin; the first page offers an
; all-users install instead.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesEnvironment=yes
SetupIconFile={#Root}\assets\icon.ico
UninstallDisplayIcon={app}\img2gcode-gui.exe
OutputDir={#Root}\dist
OutputBaseFilename=img2gcode-{#Version}-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "addtopath"; Description: "Add img2gcode to PATH (use the CLI from any terminal)"
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#Root}\target\release\img2gcode.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\target\release\img2gcode-gui.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\assets\icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\configs\default.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\fonts\ReliefSingleLine-OFL.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\img2gcode"; Filename: "{app}\img2gcode-gui.exe"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\img2gcode"; Filename: "{app}\img2gcode-gui.exe"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\img2gcode-gui.exe"; Description: "{cm:LaunchProgram,img2gcode}"; Flags: nowait postinstall skipifsilent

[Code]
// PATH lives in the user's environment for a per-user install and in the
// machine's for an all-users install.
function EnvRoot: Integer;
begin
  if IsAdminInstallMode then Result := HKEY_LOCAL_MACHINE else Result := HKEY_CURRENT_USER;
end;

function EnvKey: String;
begin
  if IsAdminInstallMode then
    Result := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
  else
    Result := 'Environment';
end;

procedure AddToPath(Dir: String);
var
  Paths: String;
begin
  if not RegQueryStringValue(EnvRoot, EnvKey, 'Path', Paths) then Paths := '';
  if Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') > 0 then Exit;
  if (Paths <> '') and (Copy(Paths, Length(Paths), 1) <> ';') then Paths := Paths + ';';
  RegWriteExpandStringValue(EnvRoot, EnvKey, 'Path', Paths + Dir);
end;

procedure RemoveFromPath(Dir: String);
var
  Paths: String;
  P: Integer;
begin
  if not RegQueryStringValue(EnvRoot, EnvKey, 'Path', Paths) then Exit;
  P := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';');
  if P = 0 then Exit;
  // Dir starts at P in Paths; take one neighbouring separator with it.
  if P > 1 then Delete(Paths, P - 1, Length(Dir) + 1) else Delete(Paths, 1, Length(Dir) + 1);
  RegWriteExpandStringValue(EnvRoot, EnvKey, 'Path', Paths);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddToPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
