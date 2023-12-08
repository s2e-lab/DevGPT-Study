class Animal {
    protected String name;
    
    protected void makeSound() {
        System.out.println("Animal makes a sound.");
    }
}

class Dog extends Animal {
    public void greet() {
        System.out.println("Dog barks, says hello!");
        makeSound(); // Accessing protected method from the superclass
    }
}

public class Main {
    public static void main(String[] args) {
        Dog dog = new Dog();
        dog.name = "Buddy"; // Accessing protected variable from subclass
        dog.greet(); // Accessing public method that indirectly calls the protected method
        // dog.makeSound(); // Error: makeSound() is protected and not accessible from outside
    }
}
