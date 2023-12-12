import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;

public class AsyncQueue<T> {

    private Queue<T> values;
    private Queue<CompletableFuture<T>> resolves;
    private boolean closed;

    public AsyncQueue() {
        values = new LinkedList<>();
        resolves = new LinkedList<>();
        closed = false;
    }

    public synchronized void enqueue(T value) {
        if (closed) {
            throw new IllegalStateException("Async closed.");
        }

        if (!resolves.isEmpty()) {
            CompletableFuture<T> resolve = resolves.poll();
            resolve.complete(value);
        } else {
            values.add(value);
        }
    }

    public synchronized CompletableFuture<T> dequeue() {
        if (!values.isEmpty()) {
            T value = values.poll();
            return CompletableFuture.completedFuture(value);
        } else if (closed) {
            return CompletableFuture.completedFuture(null);
        } else {
            CompletableFuture<T> future = new CompletableFuture<>();
            resolves.add(future);
            return future;
        }
    }

    public synchronized void close() {
        while (!resolves.isEmpty()) {
            CompletableFuture<T> resolve = resolves.poll();
            resolve.complete(null);
        }
        closed = true;
    }
}
