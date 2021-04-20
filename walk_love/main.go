package main

import (
	"github.com/lxn/walk"
	. "github.com/lxn/walk/declarative"
	"log"
	"strings"
)

func main() {

	GuiInit()
}

func GuiInit()  {
	mw, err := walk.NewMainWindow()
	if err != nil {
		log.Fatal(err)
	}
	//托盘图标文件
	icon, err := walk.Resources.Image("test.jpg")
	if err != nil {
		log.Fatal(err)
	}
	ni, err := walk.NewNotifyIcon(mw)
	if err != nil {
		log.Fatal(err)
	}
	defer ni.Dispose()
	if err := ni.SetIcon(icon); err != nil {
		log.Fatal(err)
	}
	if err := ni.SetToolTip("我爱你"); err != nil {
		log.Fatal(err)
	}
	ni.MouseDown().Attach(func(x, y int, button walk.MouseButton) {
		if button != walk.LeftButton {
			M1()
		}
		if button != walk.RightButton {
			return
		}
		if err := ni.ShowCustom("Walk 任务栏通知标题","walk 任务栏通知内容", icon); err != nil {
			log.Fatal(err)
		}
	})
	exitAction := walk.NewAction()
	if err := exitAction.SetText("右键icon的菜单按钮"); err != nil {
		log.Fatal(err)
	}
	//Exit 实现的功能
	exitAction.Triggered().Attach(func() { walk.App().Exit(0) })
	if err := ni.ContextMenu().Actions().Add(exitAction); err != nil {
		log.Fatal(err)
	}
	if err := ni.SetVisible(true); err != nil {
		log.Fatal(err)
	}
	if err := ni.ShowInfo("Walk NotifyIcon Example", "Click the icon to show again."); err != nil {
		log.Fatal(err)
	}
	mw.Run()
}

func M1() {
	var inTE, outTE *walk.TextEdit

	_, _ = MainWindow{
		Title:   "勇气",
		MinSize: Size{1080, 1080},
		Layout:  VBox{},
		Children: []Widget{
			HSplitter{
				Children: []Widget{
					TextEdit{AssignTo: &inTE, MaxLength: 10},
					TextEdit{AssignTo: &outTE, ReadOnly: true},
				},
			},
			PushButton{
				Text: "我喜欢你",
				OnClicked: func() {
					_ = outTE.SetText(strings.ToUpper(inTE.Text()))
				},
			},
		},
	}.Run()
}
