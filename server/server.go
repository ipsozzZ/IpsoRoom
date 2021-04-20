package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
)

func main() {
	listener, err := net.Listen("tcp", "localhost:8888")
	if err != nil {
		//log.Fatal()打印错误信息并调用os.Exit(1)，终止程序
		log.Fatal(err)
	}

	//广播，发送消息到所有客户端
	go broadcaster()

	for {
		conn, err := listener.Accept()
		if err != nil {
			log.Println(err)
			continue
		}
		//每个客户端一个goroutine
		go handleConn(conn)
	}
}

//channel的三种类型(只发送、只接受、即发送也接受)
//这里client只发送不接受
//只接受 type client <-chan string
//即发送也接受 type client chan string
type client chan<- string

var (
	entering = make(chan client)
	leaving  = make(chan client)
	message  = make(chan string)
)

func broadcaster() {
	clients := make(map[client]bool)
	for {
		select {
		case msg := <-message:
			for cli := range clients {
				//这里的cli就是handleConn里的ch channel，
				//writeToCLient goroutine一直在监听ch channel，读取channel中的内容，并写入客户端连接
				cli <- msg
			}
		case cli := <-entering:
			clients[cli] = true
		case cli := <-leaving:
			delete(clients, cli)
			close(cli)
		}
	}
}

func handleConn(conn net.Conn) {
	ch := make(chan string)
	//写入消息到客户端的连接
	go writeToCLient(conn, ch)

	who := conn.RemoteAddr().String()
	//当客户端连接过来时，给客户端一条消息
	//注意，这时的ch会立马被writeToCLient goroutine读取，并发送到当前客户端
	//所以已连接的其他客户端不会接受到该条消息
	ch <- "You are " + who
	//这里的message channel会被broadcaster读取，广播给所有已连接的客户端
	//注意，这时当前客户端还没给entering，所以当前客户端不会接受到该条消息
	message <- who + " 进入聊天室"
	//将当前客户端发送给entering channel，broadcaster会将当前客户端添加到已连接的客户端集合中
	entering <- ch

	input := bufio.NewScanner(conn)
	//阻塞监听客户端输入
	for input.Scan() {
		//获取客户端输入，并发送到message channel，然后broadcaster会将它广播给所有连接的客户端
		//因为这时，当前客户端已经添加到clients集合中，所以当前客户端也会接受到消息
		message <- who + ": " + input.Text()
	}

	//客户端断开连接
	leaving <- ch
	message <- who + " 离开聊天室"
	conn.Close()
}

func writeToCLient(conn net.Conn, ch <-chan string) {
	for msg := range ch {
		fmt.Fprintln(conn, msg)
	}
}
