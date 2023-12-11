import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.time.format.DateTimeFormatterBuilder;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class Main {
    private static Scanner scanner = new Scanner(System.in);
    private static EmployeeService employeeService = new EmployeeService();

    static class EmployeeService {
        private List<Employee> employees = new ArrayList<>();

        private void addEmployee(Employee employee) {
            employees.add(employee);
        }

        public void updateEmployee(String employeeId, Employee updatedEmployee) {
            Employee employee = getEmployee(employeeId);
            if (employee != null) {
                employee.setFirstName(updatedEmployee.getFirstName());
                employee.setLastName(updatedEmployee.getLastName());
                employee.setDateOfEmployment(updatedEmployee.getDateOfEmployment());
                employee.setSalary(updatedEmployee.getSalary());
                employee.setDepartment(updatedEmployee.getDepartment());
            }
        }

        private Employee getEmployee(String employeeId) {
            for (Employee employee : employees) {
                if (employee.getEmployeeId().equals(employeeId)) {
                    return employee;
                }
            }
            return null;
        }

        public void deleteEmployee(String employeeId) {
            Employee employee = getEmployee(employeeId);
            if (employee != null) {
                employees.remove(employee);
            }
        }

        public List<Employee> getAllEmployees() {
            return employees;
        }
    }

    private void addEmployee() {

        System.out.println("Enter the employee's first name:");
        String firstName = scanner.nextLine();

        System.out.println("Enter the employee's last name:");
        String lastName = scanner.nextLine();

        System.out.println("Enter the employee ID:");
        String employeeId = scanner.nextLine();

        System.out.println("Enter the employment date (dd-MM-yyyy):");
        String date = scanner.nextLine();
        LocalDate dateOfEmployment = LocalDate.parse(date, new SimpleDateFormat("dd-MM-yyyy"));

        System.out.println("Enter the salary:");
        double salary = scanner.nextDouble();
        scanner.nextLine(); // consume newline left-over

        System.out.println("Enter the department:");
        String department = scanner.nextLine();

        Employee newEmployee = new Employee(firstName, lastName, employeeId, dateOfEmployment, salary, department);
        employeeService.addEmployee(newEmployee);

        System.out.println("Employee added successfully!");
    }

    private void updateEmployee() {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the employee ID of the employee to update:");
        String employeeId = scanner.nextLine();

        System.out.println("Enter the new first name:");
        String firstName = scanner.nextLine();

        System.out.println("Enter the new last name:");
        String lastName = scanner.nextLine();

        System.out.println("Enter the new employment date (dd-MM-yyyy):");
        String date = scanner.nextLine();
        LocalDate dateOfEmployment = LocalDate.parse(date , )//.parse(date, formatter);

        System.out.println("Enter the new salary:");
        double salary = scanner.nextDouble();
        scanner.nextLine(); // consume newline left-over

        System.out.println("Enter the new department:");
        String department = scanner.nextLine();

        Employee updatedEmployee = new Employee(firstName, lastName, employeeId, dateOfEmployment, salary, department);
        employeeService.updateEmployee(employeeId, updatedEmployee);

        System.out.println("Employee updated successfully!");
    }

    private void deleteEmployee() {
        System.out.println("Enter the employee ID of the employee to delete:");
        String employeeId = scanner.nextLine();

        employeeService.deleteEmployee(employeeId);

        System.out.println("Employee deleted successfully!");
    }

    private void showEmployee() {
        System.out.println("Enter the employee ID of the employee to show:");
        String employeeId = scanner.nextLine();

        Employee employee = employeeService.getEmployee(employeeId);

        if (employee != null) {
            System.out.println("First Name: " + employee.getFirstName());
            System.out.println("Last Name: " + employee.getLastName());
            System.out.println("Employee ID: " + employee.getEmployeeId());
            System.out.println("Date of Employment: " + employee.getDateOfEmployment().format(formatter));
            System.out.println("Salary: " + employee.getSalary());
            System.out.println("Department: " + employee.getDepartment());
        } else {
            System.out.println("Employee not found!");
        }
    }

    private void showAllEmployees() {
        List<Employee> employees = employeeService.getAllEmployees();

        for (Employee employee : employees) {
            System.out.println("------------------------------");
            System.out.println("First Name: " + employee.getFirstName());
            System.out.println("Last Name: " + employee.getLastName());
            System.out.println("Employee ID: " + employee.getEmployeeId());
            System.out.println("Date of Employment: " + employee.getDateOfEmployment().format(formatter));
            System.out.println("Salary: " + employee.getSalary());
            System.out.println("Department: " + employee.getDepartment());
        }
        System.out.println("------------------------------");
    }


    class Employee {
        private String firstName;
        private String lastName;
        private String employeeId;
        private LocalDate dateOfEmployment;
        private double salary;
        private String department;

        public Employee(String firstName, String lastName, String employeeId, LocalDate dateOfEmployment, double salary, String department) {
            this.firstName = firstName;
            this.lastName = lastName;
            this.employeeId = employeeId;
            this.dateOfEmployment = dateOfEmployment;
            this.salary = salary;
            this.department = department;
        }

        public String getFirstName() {
            return firstName;
        }

        public String getLastName() {
            return lastName;
        }

        public String getEmployeeId() {
            return employeeId;
        }

        public LocalDate getDateOfEmployment() {
            return dateOfEmployment;
        }

        public double getSalary() {
            return salary;
        }

        public String getDepartment() {
            return department;
        }

        public void setFirstName(String firstName) {
            this.firstName = firstName;
        }

        public void setLastName(String lastName) {
            this.lastName = lastName;
        }

        public void setEmployeeId(String employeeId) {
            this.employeeId = employeeId;
        }

        public void setDateOfEmployment(LocalDate dateOfEmployment) {
            this.dateOfEmployment = dateOfEmployment;
        }

        public void setSalary(double salary) {
            this.salary = salary;
        }

        public void setDepartment(String department) {
            this.department = department;
        }
    }
}