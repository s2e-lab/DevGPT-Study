public class Main {
    public static void main(String[] args) {
        AsyncQueue<String> queue = new AsyncQueue<>();

        new Thread(() -> {
            queue.enqueue("Hello");
            queue.enqueue("World");
            queue.close();
        }).start();

        while (true) {
            CompletableFuture<String> future = queue.dequeue();
            if (future.isDone()) {
                try {
                    String value = future.get();
                    if (value == null) {
                        break;
                    }
                    // 处理从队列中获取的值
                    System.out.println(value);
                } catch (Exception e) {
                    // 处理异常
                    break;
                }
            }
        }
    }
}
