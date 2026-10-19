//go:build windows

package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"syscall"
	"unicode/utf16"
)

func powershell(ctx context.Context, script string, input any) ([]byte, error) {
	prefix := `[Console]::InputEncoding=New-Object System.Text.UTF8Encoding($false);[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);$ErrorActionPreference='Stop';try { `
	suffix := ` } catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }`
	codes := utf16.Encode([]rune(prefix + script + suffix))
	raw := make([]byte, len(codes)*2)
	for i, c := range codes {
		binary.LittleEndian.PutUint16(raw[i*2:], c)
	}
	cmd := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", base64.StdEncoding.EncodeToString(raw))
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	b, e := json.Marshal(input)
	if e != nil {
		return nil, e
	}
	cmd.Stdin = bytes.NewReader(b)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, e := cmd.Output()
	if e != nil {
		return nil, fmt.Errorf("Windows 語音無法執行：%s", strings.TrimSpace(stderr.String()))
	}
	return bytes.TrimPrefix(bytes.TrimSpace(out), []byte{0xef, 0xbb, 0xbf}), nil
}
func installedVoices(ctx context.Context) ([]Voice, error) {
	b, e := powershell(ctx, `Add-Type -AssemblyName System.Speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;$rows=@($s.GetInstalledVoices() | Where-Object {$_.Enabled} | ForEach-Object { @{name=$_.VoiceInfo.Name;lang=$_.VoiceInfo.Culture.Name} });ConvertTo-Json -InputObject $rows -Compress;$s.Dispose()`, nil)
	if e != nil {
		return nil, e
	}
	var v []Voice
	e = json.Unmarshal(b, &v)
	return v, e
}
func synthesize(ctx context.Context, text, voice string, rate int, path string) error {
	timingScript := `$j=([Console]::In.ReadToEnd() | ConvertFrom-Json);Add-Type -AssemblyName System.Speech;$asm=[System.Speech.Synthesis.SpeechSynthesizer].Assembly.Location;Add-Type -ReferencedAssemblies $asm -TypeDefinition @'
using System;
using System.IO;
using System.Text;
using System.Globalization;
using System.Speech.Synthesis;
public class JunbaTimedSpeech {
 public static void Render(string text,string voice,int rate,string path) {
  using(var s=new SpeechSynthesizer()) {
   if(!String.IsNullOrEmpty(voice)) s.SelectVoice(voice);
   else { bool found=false;foreach(var v in s.GetInstalledVoices()){if(v.Enabled && v.VoiceInfo.Culture.Name.StartsWith("zh",StringComparison.OrdinalIgnoreCase)){s.SelectVoice(v.VoiceInfo.Name);found=true;break;}}if(!found)throw new Exception("找不到中文桌面語音，請安裝中文語音或匯入 WAV。"); }
   using(var writer=new StreamWriter(path+".timing.tsv",false,new UTF8Encoding(false))) {
    bool capturing=true;
    s.SpeakProgress+=(sender,e)=>{lock(writer){if(!capturing)return;writer.WriteLine(e.CharacterPosition.ToString(CultureInfo.InvariantCulture)+"\t"+e.AudioPosition.TotalSeconds.ToString("F4",CultureInfo.InvariantCulture));writer.Flush();}};
    s.Rate=rate;s.SetOutputToWaveFile(path);s.Speak(text);s.SetOutputToNull();lock(writer){capturing=false;}
   }
  }
 }
}
'@;[JunbaTimedSpeech]::Render([string]$j.text,[string]$j.voice,[int]$j.rate,[string]$j.path)`
	if _, err := powershell(ctx, timingScript, map[string]any{"text": text, "voice": voice, "rate": rate, "path": path}); err == nil {
		return nil
	}
	_, e := powershell(ctx, `$j=([Console]::In.ReadToEnd() | ConvertFrom-Json);Add-Type -AssemblyName System.Speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;try {if($j.voice){$s.SelectVoice([string]$j.voice)}else{$v=$s.GetInstalledVoices() | Where-Object {$_.Enabled -and $_.VoiceInfo.Culture.Name -like 'zh*'} | Select-Object -First 1;if(-not $v){throw '找不到中文桌面語音。請在 Windows 安裝中文語音，或改匯入 WAV 錄音。'};$s.SelectVoice($v.VoiceInfo.Name)};$s.Rate=[int]$j.rate;$s.SetOutputToWaveFile([string]$j.path);$s.Speak([string]$j.text);$s.SetOutputToNull()}finally{$s.Dispose()}`, map[string]any{"text": text, "voice": voice, "rate": rate, "path": path})
	return e
}
func openBrowser(url string) error {
	c := exec.Command("rundll32.exe", "url.dll,FileProtocolHandler", url)
	c.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	return c.Start()
}
func openFolder(path string) error { return exec.Command("explorer.exe", path).Start() }
func hideCommand(cmd *exec.Cmd)    { cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true} }
