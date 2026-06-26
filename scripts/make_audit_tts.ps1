$Text = @"
Teacher: Today we are reviewing linear equations. First, solve two x plus three equals eleven.
Student: I subtract three and get two x equals eight. Then I divide by two, so x equals four.
Teacher: Good. Now try five minus x equals two x minus one.
Student: I move x to the right and minus one to the left. Six equals three x, so x equals two.
Teacher: Correct. The main thing is to change signs carefully when moving terms.
Student: I sometimes forget the sign, especially when there is a minus before x.
Teacher: For homework, solve five similar equations and write one line explaining each transformation.
"@

Add-Type -AssemblyName System.Speech
$Synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
$Synthesizer.SelectVoice("Microsoft Zira Desktop")
$Synthesizer.Rate = -1
$Synthesizer.Volume = 100
$OutputPath = Join-Path $env:USERPROFILE "Downloads\EasyRepet_Audit_Simulated_Lesson_20260626.wav"
$Synthesizer.SetOutputToWaveFile($OutputPath)
$Synthesizer.Speak($Text)
$Synthesizer.Dispose()
Get-Item -LiteralPath $OutputPath | Select-Object FullName,Length,LastWriteTime
