class Car {
    private String brand;
    private int year;

    public Car(String brand, int year) {
        this.brand = brand;
        this.year = year;
    }

    public String getBrand() {
        return brand;
    }

    public int getYear() {
        return year;
    }

    public void start() {
        System.out.println("Car started.");
    }

    private void accelerate() {
        System.out.println("Car is accelerating.");
    }
}

public class Main {
    public static void main(String[] args) {
        Car myCar = new Car("Toyota", 2023);
        
        // Accessing attributes using getter methods
        System.out.println("Brand: " + myCar.getBrand());
        System.out.println("Year: " + myCar.getYear());

        // Calling public methods
        myCar.start();

        // Cannot access private method directly
        // myCar.accelerate(); // Compile-time error

        // Cannot directly access or modify private attributes
        // myCar.brand = "Honda"; // Compile-time error
        // int carYear = myCar.year; // Compile-time error
    }
}
