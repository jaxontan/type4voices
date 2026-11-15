' Starts Voice Dictation in the background without a console window.
' All output goes to dictate.log, so nothing is lost when there is no console.
Option Explicit
Dim fso, shell, appDir, pythonw, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

appDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = appDir & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(pythonw) Then
    MsgBox "Python environment not found." & vbCrLf & _
           "Expected: " & pythonw & vbCrLf & vbCrLf & _
           "Run setup.bat once, then try again.", 16, "Voice Dictation"
    WScript.Quit 1
End If

shell.CurrentDirectory = appDir
cmd = """" & pythonw & """ """ & appDir & "\dictate.py"""
shell.Run cmd, 0, False
