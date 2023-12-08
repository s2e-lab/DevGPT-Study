import { render, screen, fireEvent } from "@testing-library/react";
import App from "./App";

test("renders header Text", () => {
	render(<App />);
	const linkElement = screen.getByText(
		/Auto generate personalised email sequences/i
	);
	expect(linkElement).toBeInTheDocument();
});

test("renders textarea placeholder", () => {
	render(<App />);
	const textareaElement = screen.getByPlaceholderText(
		/Write the email prompt/i
	);
	expect(textareaElement).toBeInTheDocument();
});

test("updates email prompt on textarea change", () => {
	render(<App />);
	const textareaElement = screen.getByPlaceholderText(
		/Write the email prompt/i
	);

	fireEvent.change(textareaElement, {
		target: { value: "New email prompt" },
	});

	expect(textareaElement.value).toBe("New email prompt");
});
