package main

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseDialogueScriptAndTags(t *testing.T) {
	lines, e := parseDialogueScript("A[nod, sway]：大家好，我是峻爸！\n\nＢ[側傾]: 你好！\nc：今天天氣真好\nA: 下次見")
	if e != nil {
		t.Fatal(e)
	}
	if len(lines) != 4 || lines[0].Speaker != "A" || lines[1].Speaker != "B" || lines[2].Speaker != "C" || lines[3].Speaker != "A" {
		t.Fatalf("speakers %+v", lines)
	}
	if strings.Join(lines[0].Actions, ",") != "nod,sway" || strings.Join(lines[1].Actions, ",") != "tilt" || len(lines[2].Actions) != 0 {
		t.Fatalf("actions %+v", lines)
	}
	if lines[0].HanText != "大家好，我是峻爸！" {
		t.Fatal(lines[0].HanText)
	}
	for _, bad := range []string{"沒有說話者", "F：超過五人", "A："} {
		if _, e := parseDialogueScript(bad); e == nil {
			t.Fatal("accepted", bad)
		}
	}
}

func TestParseDialogueTimeline(t *testing.T) {
	tl, e := parseDialogueTimeline("A 0-3.2\nB 3.4～6 秒\na 6.2-9.5", 8)
	if e != nil || len(tl) != 3 || tl[1].Speaker != "B" || tl[1].Start != 3.4 || tl[2].End != 8 {
		t.Fatalf("%v %+v", e, tl)
	}
	if _, e := parseDialogueTimeline("A 3-1", 8); e == nil {
		t.Fatal("reversed range accepted")
	}
}

func TestValidateSpeakers(t *testing.T) {
	ok := []SpeakerBox{{ID: "A", MouthBox: []float64{.1, .5, .2, .05}}, {ID: "B", MouthBox: []float64{.6, .5, .2, .05}, FaceBox: []float64{.55, .2, .3, .4}}}
	if e := validateSpeakers(ok); e != nil {
		t.Fatal(e)
	}
	if validateSpeakers(append(ok, SpeakerBox{ID: "B", MouthBox: []float64{.1, .1, .1, .1}})) == nil {
		t.Fatal("duplicate speaker accepted")
	}
	if validateSpeakers([]SpeakerBox{{ID: "F", MouthBox: []float64{.1, .1, .1, .1}}}) == nil {
		t.Fatal("speaker F accepted")
	}
	six := []SpeakerBox{}
	for _, id := range []string{"A", "B", "C", "D", "E", "A"} {
		six = append(six, SpeakerBox{ID: id, MouthBox: []float64{.1, .1, .1, .1}})
	}
	if validateSpeakers(six) == nil {
		t.Fatal("six speakers accepted")
	}
}

func TestDialogueCaptionsTwoLines(t *testing.T) {
	dir := t.TempDir()
	d := Dialogue{Names: map[string]string{"A": "峻爸"}, Lines: []DialogueLine{
		{Speaker: "A", HanText: "大家好，我是峻爸！今天一起嘗試新的方向。", Start: 0, End: 3},
		{Speaker: "B", HanText: "你好！", TailoText: "Lí hó!", Start: 3.3, End: 4.5},
	}}
	ok, e := writeDialogueCaptions(dir, d, 5)
	if e != nil || !ok {
		t.Fatal(e)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "subtitles.srt"))
	srt := string(b)
	if !strings.Contains(srt, "峻爸：大家好，") || !strings.Contains(srt, "B：你好！\nLí hó!") || !strings.Contains(srt, "00:00:03,300 --> 00:00:04,500") {
		t.Fatal(srt)
	}
}

func TestSingleSpeakerSecondCaptionLine(t *testing.T) {
	dir := t.TempDir()
	ok, timing, e := writeCaptionsSecond(dir, "大家好，我是峻爸！\n今天天氣真好。", "Tāi-ke hó, guá sī Tsùn-pē!\nKin-á-jit thinn-khì tsin hó.", 6)
	if e != nil || !ok || !strings.Contains(timing, "雙行") {
		t.Fatal(e, timing)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "subtitles.srt"))
	if !strings.Contains(string(b), "大家好，我是峻爸！\nTāi-ke hó, guá sī Tsùn-pē!") || strings.Count(string(b), " --> ") != 2 {
		t.Fatal(string(b))
	}
}

func TestPrepareDialogueWithPerSpeakerVoices(t *testing.T) {
	if _, e := exec.LookPath("ffmpeg"); e != nil {
		t.Skip("ffmpeg required")
	}
	a := newApp(t.TempDir())
	var calls []string
	a.tts = func(ctx context.Context, text, voice string, rate int, path string) error {
		calls = append(calls, voice+"|"+text)
		testWAV(t, path, 2)
		return nil
	}
	var body bytes.Buffer
	m := multipart.NewWriter(&body)
	m.WriteField("text", "A[nod]：大家好！\nB[tilt]：你好！\nA：再見！")
	m.WriteField("dialogue", "1")
	m.WriteField("voice_A", "VoiceA")
	m.WriteField("voice_B", "VoiceB")
	m.WriteField("effect_B", "cartoon")
	m.WriteField("speaker_name_A", "峻爸")
	m.WriteField("caption_second", "Tāi-ke hó!\nLí hó!\nTsài-kiàn!")
	m.Close()
	w := request(t, a, "POST", "/api/prepare", &body, m.FormDataContentType(), true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	var j Job
	json.Unmarshal(w.Body.Bytes(), &j)
	if !j.Dialogue || strings.Join(j.DialogueSpeakers, ",") != "A,B" || !j.HasCaptions {
		t.Fatalf("%+v", j)
	}
	if strings.Join(calls, ";") != "VoiceA|大家好！;VoiceB|你好！;VoiceA|再見！" {
		t.Fatal(calls)
	}
	var d Dialogue
	b, _ := os.ReadFile(filepath.Join(a.dir(j.ID), "dialogue.json"))
	if json.Unmarshal(b, &d) != nil || len(d.Lines) != 3 || d.Lines[1].Start <= d.Lines[0].End || d.Lines[2].End < 6 || d.Lines[1].TailoText != "Lí hó!" || d.Lines[0].Actions[0] != "nod" {
		t.Fatalf("%+v", d)
	}
	if j.Duration < 6 || j.Duration > 7.2 {
		t.Fatal("duration", j.Duration)
	}
}
