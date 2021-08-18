package likou

import (
	"math"
	"strings"
	"testing"
)

func Test94(t *testing.T) {
	//res := inorderTraversal(&TreeNode{Val:1, Left:nil, Right:&TreeNode{Val:2, Left:&TreeNode{Val:3}, Right:nil}})
	//t.Log(res)

	res := inorderTraversal(&TreeNode{})
	t.Log(res)
}




type TreeNode struct {
	Val int
    Left *TreeNode
	Right *TreeNode
}

func inorderTraversal(root *TreeNode) (res []int) {
	var inorder func(node *TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		res = append(res, node.Val)
		inorder(node.Right)
	}

	inorder(root)
	return
}


// 递归
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return 1 + int(math.Max(float64(maxDepth(root.Left)), float64(maxDepth(root.Right))))
}

// 非递归
func maxDepthBFS(root *TreeNode) int {
	if root == nil {
		return 0
	}

	var queue []*TreeNode
	queue = append(queue, root)
	depth := 0

	for len(queue) > 0 {
		size := len(queue)

		for i:=0; i<size; i++{
			s := queue[0]
			queue = queue[1:]
			if s.Left != nil {
				queue = append(queue, s.Left)
			}
			if s.Right != nil {
				queue = append(queue, s.Right)
			}
		}
		depth++
	}
	return depth
}

func TestMaxDepth(t *testing.T) {
	tree := &TreeNode{Val: 0, Left:&TreeNode{Val:1, Right: &TreeNode{Val:2}}, Right:&TreeNode{Val:3, Right:&TreeNode{Val:4, Left:&TreeNode{Val:5, Left:&TreeNode{Val:6}}}}}
	depth := maxDepthBFS(tree)
	t.Logf("tree max depth is %d \n", depth)
}

// 将有序数组转平衡二叉树
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {return nil}
	root := &TreeNode{nums[len(nums)/2], nil, nil}
	root.Left  = sortedArrayToBST(nums[:len(nums)/2])
	root.Right = sortedArrayToBST(nums[len(nums)/2+1:])
	return root
}

func TestSortedArrayToBST(t *testing.T) {
	nums := []int{-10,-3,0,5,9}
	blt := sortedArrayToBST(nums)
	t.Logf("tree max depth is %+v \n", blt)
}




// 110 平衡二叉树
func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	if !isBalanced(root.Left) || !isBalanced(root.Right) {
		return false
	}
	lh := maxDepth1(root.Left)+1
	rh := maxDepth1(root.Right)+1

	if math.Abs(float64(lh-rh)) > 1 {
		return false
	}
	return true
}

func maxDepth1(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return int(math.Max(float64(maxDepth1(root.Left)), float64(maxDepth1(root.Right)))) + 1
}



/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */

type ListNode struct {
	Val int
	Next *ListNode
}

func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	next := head.Next
	head.Next = swapPairs(next.Next)
	next.Next = head
	return next
}

func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right != nil {
		return 1 + minDepth(root.Right)
	}
	if root.Left != nil && root.Right == nil {
		return 1 + minDepth(root.Left)
	}
	return isSmall(minDepth(root.Left), minDepth(root.Right)) + 1
}

func isSmall(a, b int) int {
	if a > b {
		return b
	}
	return a
}


// 112 路径总和
func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	if targetSum - root.Val == 0 && root.Right == nil && root.Left == nil  {
		return true
	}
	return hasPathSum(root.Right, targetSum - root.Val) || hasPathSum(root.Left, targetSum - root.Val)
}

// 53 最大子序和
func maxSubArray(nums []int) int {
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] + nums[i-1] > nums[i] {
			nums[i] += nums[i-1]
		}
		if nums[i] > res {
			res = nums[i]
		}
	}
	return res
}


// 70 爬楼梯
func climbStairs(n int) int {
	if n < 3 {
		return n
	}
	stp1 := 1
	stp2 := 2
	for i := 3; i<=n; i++ {
		temp := stp1 + stp2
		stp1 = stp2
		stp2 = temp
	}
	return stp2
}

func TestDp70(t *testing.T) {
	t.Logf("爬楼梯 === %d", climbStairs(3))
}


// 杨辉三角

/**

1
1 2
1 3 1
1 4 4 1
1 5 8 5 1
1 6 13 13 6 1

*/
func generate(numRows int) [][]int {
	res := make([][]int, numRows)
	for i := range res {
		res[i] = make([] int, i+1)
		res[i][0] = 1
		res[i][i] = 1
		for j := 1; j < i; j++ {
			res[i][j] = res[i-1][j] + res[i-1][j-1]
		}
	}
	return res
}

func Test(t *testing.T) {
	t.Logf("generate == %+v", generate(9))
}



// 旋转数组

func rotate(nums []int, k int) {
	n := len(nums)
	k %= n
	for start, count := 0, gcd(k, n); start < count; start++ {
		pre, cur := nums[start], start
		for ok := true; ok; ok = cur != start {
			next := (cur + k) % n
			nums[next], pre, cur = pre, nums[next], next
		}
	}
}

func gcd(a, b int) int {
	for a != 0 {
		a, b = b%a, a
	}
	return b
}

func rotate1(nums []int, k int) {
	newNums := make([]int, len(nums))
	for i, v := range nums {
		newNums[(i+k)%len(nums)] = v
	}
	copy(nums, newNums)
}

func reverse(a []int) {
	for i, n := 0, len(a); i < n/2; i++ {
		a[i], a[n-1-i] = a[n-1-i], a[i]
	}
}

func rotate2(nums []int, k int) {
	k %= len(nums)
	reverse(nums)
	reverse(nums[:k])
	reverse(nums[k:])
}

// 两数之和II - 输入有序数组  双指针
func twoSum(numbers []int, target int) []int {
	begin, end := 0, len(numbers) - 1
	for begin < end {
		sum := numbers[begin] + numbers[end]
		if sum == target {
			return []int{begin+1, end+1}
		}else if sum > target {
			end--
		}else {
			begin++
		}
	}
	return []int{-1, -1}
}


//  两数之和II - 输入有序数组  二分查找
func twoSum1(numbers []int, target int) []int {
	nl := len(numbers)
	for i:=0; i<nl; i++ {
		low := i
		high := nl - 1
		for low < high {

		}
	}
	return []int{-1, -1}
}



// 19 删除链表的倒数第n个节点
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	nodes := []*ListNode{}
	res := &ListNode{0, head}
	for node := res; node != nil; node = node.Next {
		nodes = append(nodes, node)
	}
	Nnode := nodes[len(nodes)-1-n]
	Nnode.Next = Nnode.Next.Next
	return res.Next
}

func removeNthFromEnd1(head *ListNode, n int) *ListNode {
	hl := 0
	t := head
	for ;t != nil ; t = t.Next {
		hl++
	}

	res := &ListNode{0, head}
	curr := res
	for i := 0; i < hl-n; i++ {
		curr = curr.Next
	}
	curr.Next = curr.Next.Next
	return res.Next
}

// 3 无重复的字符的最长子序列
func lengthOfLongestSubstring(s string) int {
	// 哈希集合，记录每个字符是否出现过
	m := map[byte]int{}
	n := len(s)
	// 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
	rk, ans := -1, 0
	for i := 0; i < n; i++ {
		if i != 0 {
			// 左指针向右移动一格，移除一个字符
			delete(m, s[i-1])
		}
		for rk + 1 < n && m[s[rk+1]] == 0 {
			// 不断地移动右指针
			m[s[rk+1]]++
			rk++
		}
		// 第 i 到 rk 个字符是一个极长的无重复字符子串
		ans = max(ans, rk - i + 1)
	}
	return ans
}

func max(x, y int) int {
	if x < y {
		return y
	}
	return x
}

// 初始赋值start,end 为0
//如果end 没出现过，则end++,同时尝试更新最大子串长度
//如果end出现过，则把start移动到之前出现的index+1的位置，此后start到end之间无重复元素，继续右移end
//关键点不要用map，使用数组直接查找索引节省时间和内存
func lengthOfLongestSubstring1(s string) int {
	if len(s) == 0 {
		return 0
	}

	start := 0
	end := 0
	var appeared [256]int
	for i := 0; i < len(appeared); i++ {
		appeared[i] = -1
	}
	max := 0
	for end < len(s) {
		if idx := appeared[s[end] - 'a']; idx != -1 && idx >= start {
			start = idx + 1
		}
		tmp := end - start + 1
		if tmp > max {
			max = tmp
		}
		appeared[s[end]-'a'] = end
		end++
	}
	return max
}

// 核心：只增大不减小的滑动窗口
//流程：两个指针start和end表示窗口大小，遍历一次字符串，窗口在遍历过程中滑动或增大
//tips：配合画图思考更佳
//
//窗口内没有重复字符：此时判断i+1与end的关系，超过表示遍历到窗口之外了，增大窗口大小
//窗口内出现重复字符：此时两个指针都增大index+1，滑动窗口位置到重复字符的后一位
//遍历结束，返回end-start，窗口大小
//思考：如果需要返回字符串怎么做？
//解答：只需要在窗口增大的时候记录start指针即可
func lengthOfLongestSubstring2(s string) int {
	start, end := 0, 0
	for i := 0; i < len(s); i++ {
		index := strings.Index(s[start:i], string(s[i]))
		if index == -1 {
			if i+1 > end {
				end = i + 1
			}
		} else {
			start += index + 1
			end += index + 1
		}
	}
	return end - start
}

// 567 字符串的排列  (未成功)
func checkInclusion(s1 string, s2 string) bool {
	start, end := 0,0
	sl1 := len(s1)
	sl2 := len(s2)
	for i := 0; i<sl2;i++ {
		index := strings.Index(s1, string(s2[i]))
		if index != -1 {
			if i+1 > end {
				end = i + 1
			}
		} else {
			start += index + 1
			end += index + 1
		}
		if end - start >= sl1 {
			return true
		}
	}
	return false
}

// 用数组代替map可以比较
// 比较相同长度的s1和s2的子串对应的数组是否相等来判断是否为子串
// s2的子串通过滑动窗口来移动
func checkInclusion1(s1, s2 string) bool {
	if len(s1) > len(s2) {
		return false
	}
	cnt1, cnt2 := [26]int{}, [26]int{}
	for i := range s1 {
		cnt1[s1[i]-'a']++
		cnt2[s2[i]-'a']++
	}
	for i := 0; i < len(s2)-len(s1); i++ {
		if cnt1 == cnt2 {
			return true
		}
		cnt2[s2[i]-'a']--
		cnt2[s2[i+len(s1)]-'a']++
	}
	return cnt1 == cnt2
}


func Test567(t *testing.T) {
	cnt1, cnt2 := [26]int{}, [26]int{}
	s1 := "ab"
	s2 := "eidbaooo"
	for i := range s1 {
		cnt1[s1[i]-'a']++
		cnt2[s2[i]-'a']++
		cnt2[s2[i]-'a']++
	}
	t.Logf("cnt1 == %+v", cnt1)
	t.Logf("cnt2 == %+v", cnt2)
	t.Logf("cnt2 == cnt1 %v", cnt1==cnt2)
}


// 733 图像渲染
// 广度优先搜索
func floodFill(image [][]int, sr int, sc int, newColor int) [][]int {
	oldColor := image[sr][sc]
	if oldColor == newColor {
		return image
	}
	lx, ly := len(image), len(image[0])
	qeue := [][]int{{sr, sc}}
	image[sr][sc] = newColor
	for p := 0; p < len(qeue); p++ {
		cell := qeue[p]
		for i := 0; i < 4; i++ {
			cx,cy := cell[0]+dx[i],cell[1]+dy[i]
			if cx >= 0 && cx < lx && cy >= 0 && cy < ly && image[cx][cy] == oldColor {
				qeue = append(qeue, []int{cx,cy})
				image[cx][cy] = newColor
			}
		}
	}
	return image
}

var (
	dx = []int{1, 0, 0, -1}
	dy = []int{0, 1, -1, 0}
)

// 深度优先搜索
func floodFill1(image [][]int, sr int, sc int, newColor int) [][]int {
	currColor := image[sr][sc]
	if currColor != newColor {
		dfs(image, sr, sc, currColor, newColor)
	}
	return image
}
func dfs(image [][]int, x, y, color, newColor int) {
	if image[x][y] == color {
		image[x][y] = newColor
		for i := 0; i < 4; i++ {
			mx, my := x + dx[i], y + dy[i]
			if mx >= 0 && mx < len(image) && my >= 0 && my < len(image[0]) {
				dfs(image, mx, my, color, newColor)
			}
		}
	}
}



// 深度优先搜索
func maxAreaOfIsland(grid [][]int) int {
	max_area := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == 1 {
				max_area = max(max_area, dfs656(grid, i, j))
			}
		}
	}
	return max_area
}
func dfs656(grid [][]int, i, j int) int {
	if i < 0 || j < 0 || i >= len(grid) || j >= len(grid[0]) || grid[i][j] == 0 {
		return 0
	}
	area := 1
	grid[i][j] = 0
	area += dfs656(grid, i+1, j)
	area += dfs656(grid, i-1, j)
	area += dfs656(grid, i, j+1)
	area += dfs656(grid, i, j-1)
	return area
}
func max1(x, y int) int {
	if x > y {
		return x
	}
	return y
}


// 合并二叉树
func mergeTrees(t1, t2 *TreeNode) *TreeNode {
	if t1 == nil {
		return t2
	}
	if t2 == nil {
		return t1
	}
	t1.Val += t2.Val
	t1.Left = mergeTrees(t1.Left, t2.Left)
	t1.Right = mergeTrees(t1.Right, t2.Right)
	return t1
}


// 116. 填充每个节点的下一个右侧节点指针
type Node struct {
     Val int
     Left *Node
     Right *Node
     Next *Node
}

func connect(root *Node) *Node {
	if root == nil {
		return root
	}

	// 每次循环从该层的最左侧节点开始
	for leftmost := root; leftmost.Left != nil; leftmost = leftmost.Left {
		// 通过 Next 遍历这一层节点，为下一层的节点更新 Next 指针
		for node := leftmost; node != nil; node = node.Next {
			// 左节点指向右节点
			node.Left.Next = node.Right

			// 右节点指向下一个左节点
			if node.Next != nil {
				node.Right.Next = node.Next.Left
			}
		}
	}

	// 返回根节点
	return root
}


func orangesRotting(grid [][]int) (res int) {
	q := [][]int{}
	fresh := 0
	for i, row := range grid {
		for j, v := range row {
			if v == 2 {
				q = append(q, []int{i, j})
			} else if v == 1 {
				fresh++
			}
		}
	}
	m, n := len(grid), len(grid[0])
	dir := [][]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for len(q) > 0 && fresh > 0 {
		tmp := [][]int{}
		for _, v := range q {
			for _, d := range dir {
				x, y := v[0]+d[0], v[1]+d[1]
				if x >= 0 && y >= 0 && x < m && y < n && grid[x][y] == 1 {
					tmp = append(tmp, []int{x, y})
					grid[x][y] = 2
					fresh--
				}
			}
		}
		q = tmp
		res++
	}
	if fresh == 0 {
		return res
	}
	return -1
}


// 21合并两个有序列表
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	if l1.Val < l2.Val {
		l1.Next = mergeTwoLists(l1.Next, l2)
		return l1
	}
	l2.Next = mergeTwoLists(l1, l2.Next)
	return  l2
}



// 206 反转列表
func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	curr := head
	for curr != nil  {
		next := curr.Next
		curr.Next = pre
		pre = curr
		curr = next
	}
	return pre
}


func reverseList1(head *ListNode) *ListNode {
	return reverseLst(head, nil)
}

func reverseLst(head, pre *ListNode) *ListNode {
	if head == nil {
		return pre
	}
	next := head.Next
	head.Next = pre
	return reverseLst(next, head)
}


// 77 组合
func combine(n int, k int) (ans [][]int) {
	temp := []int{}
	var dfs func(int)
	dfs = func(cur int) {
		if len(temp) + (n - cur + 1) < k {
			return
		}
		if len(temp) == k {
			comb := make([]int, k)
			copy(comb, temp)
			ans = append(ans, comb)
			return
		}
		temp = append(temp, cur)
		dfs(cur + 1)
		temp = temp[:len(temp)-1]
		dfs(cur + 1)
	}
	dfs(1)
	return
}


// 46 全排列
func permute(nums []int) [][]int {
	res := [][]int{}
	visited := map[int]bool{}

	var dfs func(path []int)
	dfs = func(path []int) {
		if len(path) == len(nums) {
			temp := make([]int, len(path))
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for _, n := range nums {
			if visited[n] {
				continue
			}
			path = append(path, n)
			visited[n] = true
			dfs(path)
			path = path[:len(path)-1]
			visited[n] = false
		}
	}

	dfs([]int{})
	return res
}



// 784. 字母大小写全排列
func letterCasePermutation(S string) []string {
	var (
		ans    []string
		dfs    func(start int, path []byte)
		length = len(S)
		str    = []byte(S)
	)

	inArea := func(b byte) bool {
		return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
	}

	dfs = func(start int, sub []byte) {
		if start == length {
			ans = append(ans, string(sub))
			return
		}
		// 未修改当前字符(字母或者数字)的一条分支
		dfs(start+1, str)
		// 修改当前字母的的另一条分支
		if inArea(str[start]) {
			// 大小写转换
			str[start] ^= 32
			dfs(start+1, str)
		}
	}
	dfs(0, str)
	return ans
}

