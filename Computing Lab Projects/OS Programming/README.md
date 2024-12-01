# OS Programming Assignment (CS69011)

## Overview

This repository contains solutions for **Assignment: OS Programming** as part of the **CS69011 Computing Lab** course. The assignment involves creating a shell with advanced functionalities using Linux system calls, libraries such as `ncurses`, `readline`, `X11`, and threading via `pthreads`. All parts of the assignment (Parts A to F) are implemented in a single C file named `23CS60R78_Assgn_7.c`.

---

## Assignment Structure

```
.
├── 23CS60R78_Assgn_7.c        # Single C file containing all parts (A to F)
└── README.md                  # This file
```

---

## Requirements

1. **Linux OS**: Recommended for development and testing.
2. **Libraries and Dependencies**:
   - **System Calls**: `fork`, `exec`, `dup`, `dup2`, etc.
   - **ncurses**: For the text editor.
     - Install using: `sudo apt install libncurses5-dev libncursesw5-dev`
   - **readline**: For command history and editing.
     - Install using: `sudo apt install libreadline-dev`
   - **X11**: For GUI shell.
     - Install using: `sudo apt install libx11-dev`
   - **Pthreads**: For threading-based vector operations.
     - Typically included in standard C libraries; link with `-pthread` flag.

---

## Compilation Instructions

Compile the C program with the necessary libraries and flags:

```bash
gcc 23CS60R78_Assgn_7.c -o shell -lreadline -lncurses -lX11 -pthread
```

---

## How to Run

Run the compiled executable:

```bash
./shell
```

---

## Features and Usage

### Part A: Basic Shell

- **Commands Supported**:
  - `pwd`: Display current working directory.
  - `cd <directory>`: Change current directory.
  - `mkdir <directory>`: Create a new directory.
  - `ls <flags>`: Display directory contents.
  - `exit`: Exit the shell.
  - `help`: List supported commands.
  - **Note**: Any other command is treated as an executable file.

- **Background Execution**:
  - Append `&` at the end of a command to run it in the background.

### Part B: I/O Redirection and Piping

- **Piping**:
  - Use the `|` operator to pipe output from one command to another.
  - **Example**:
    ```bash
    ls -al | grep file
    ```

### Part C: Readline Library Enhancements

- **Multiline Commands**:
  - Use `\` at the end of a line to continue typing on the next line.
- **Command History**:
  - Use the Up and Down arrow keys to navigate through command history.
- **Command Editing**:
  - Use `Ctrl+A` to move to the start of the line.
  - Use `Ctrl+E` to move to the end of the line.
  - Left and Right arrow keys to move the cursor within the line.

### Part D: Text Editor with ncurses

- **Invoke Editor**:
  - Use `vi <filename>` to open the built-in text editor.
- **Editor Functionalities**:
  - **Cursor Movement**: Use arrow keys to navigate.
  - **Insert Characters**: Type alphanumeric keys to insert text.
  - **Delete Character**: Use the `Delete` key.
  - **Save File**: Press `Ctrl+S` to save the file.
  - **Exit Editor**: Press `Ctrl+X` to exit.
  - **Exit without Saving**: Press `ESC` to exit without saving.
- **Upon Exit**:
  - Displays the number of lines, words, and characters edited.

### Part E: GUI Shell with X11

- **Standalone GUI Application**:
  - The shell runs in its own window using X11.
- **Features**:
  - Integrates all previous shell functionalities in a graphical interface.
- **Note**:
  - Ensure X11 forwarding is enabled if running remotely.
  - May require a display server (like XQuartz on macOS).

### Part F: Vector Operations with Pthreads

- **Commands Supported**:
  - `addvec <file1> <file2> -<no_thread>`: Adds two vectors.
  - `subvec <file1> <file2> -<no_thread>`: Subtracts two vectors.
  - `dotprod <file1> <file2> -<no_thread>`: Calculates dot product.
- **Vector Files**:
  - Each file contains a single line with `n` space-separated numbers.
- **Number of Threads**:
  - `<no_thread>` specifies how many threads to use (default is 3).
- **Usage Examples**:
  ```bash
  addvec vector1.txt vector2.txt -4
  subvec vector1.txt vector2.txt
  dotprod vector1.txt vector2.txt -2
  ```
- **Output**:
  - The result of the vector operation is displayed in the shell.

---

## Error Handling

- Proper error messages are displayed if:
  - Incorrect commands are entered.
  - Files do not exist or cannot be accessed.
  - Invalid number of arguments are provided.

---

## References

1. **Pthreads Tutorial**: [LLNL Pthreads Tutorial](https://hpc-tutorials.llnl.gov/posix/)
2. **Xlib Programming**: [Xlib Manual](https://tronche.com/gui/x/xlib/)
3. **System Calls**: Use `man` pages for detailed information on system calls and functions.

---

## Compilation Notes

- **Include Flags**:
  - `-lreadline`: Links the readline library.
  - `-lncurses`: Links the ncurses library.
  - `-lX11`: Links the X11 library.
  - `-pthread`: Enables pthread support.

- **Example Compilation Command**:

  ```bash
  gcc 23CS60R78_Assgn_7.c -o shell -lreadline -lncurses -lX11 -pthread
  ```

---

## Author

**Avik Pramanick**  
**Roll No:** 23CS60R78  
**Course:** CS69011 Computing Lab  
