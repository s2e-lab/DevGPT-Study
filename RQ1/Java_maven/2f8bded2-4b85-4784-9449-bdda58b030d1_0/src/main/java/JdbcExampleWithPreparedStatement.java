import java.sql.*;

public class JdbcExampleWithPreparedStatement {
    public static void main(String[] args) {
        String jdbcUrl = "jdbc:mysql://localhost:3306/mydatabase"; // Replace 'mydatabase' with your database name
        String username = "your_username";
        String password = "your_password";

        try {
            // Step 1: Load and register the JDBC driver (MySQL)
            Class.forName("com.mysql.cj.jdbc.Driver");

            // Step 2: Establish the connection to the database
            Connection connection = DriverManager.getConnection(jdbcUrl, username, password);

            // Step 3: Create a prepared statement object to execute SQL queries
            String sqlQuery = "SELECT * FROM my_table WHERE id=?";
            PreparedStatement preparedStatement = connection.prepareStatement(sqlQuery);

            // Step 4: Set parameters for the prepared statement
            int idParameter = 1; // Replace 1 with the desired ID value you want to search for
            preparedStatement.setInt(1, idParameter);

            // Step 5: Execute the prepared statement and get the result set
            ResultSet resultSet = preparedStatement.executeQuery();

            // Step 6: Process the result set
            while (resultSet.next()) {
                // Assuming you have 'id' and 'name' columns in your table
                int id = resultSet.getInt("id");
                String name = resultSet.getString("name");

                // Print the retrieved data
                System.out.println("ID: " + id + ", Name: " + name);
            }

            // Step 7: Close the resources
            resultSet.close();
            preparedStatement.close();
            connection.close();
        } catch (ClassNotFoundException e) {
            System.out.println("JDBC driver not found!");
            e.printStackTrace();
        } catch (SQLException e) {
            System.out.println("Error executing SQL query!");
            e.printStackTrace();
        }
    }
}
