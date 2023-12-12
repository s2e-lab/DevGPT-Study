public class SampleSubscriber {

    @Observe
    public void onStringEvent(Event<String> event, String additionalData, int count) {
        System.out.println("Received: " + event.getData() + ", Additional: " + additionalData + ", Count: " + count);
    }

    // Other methods with @Observe can be added for demonstration.
}
