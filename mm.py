
class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def say_hello(self):
        print("Hello, my name is {} and I am {} years old".format(self.name, self.age)
)

# Create a new Person object
person = Person("John", 36)

# Call the say_hello method
person.say_hello()
print(person.age)
# What is the output of this code?
# Hello, my name is John and I am 36 years old
person = Person("John", "zwei")

print(person.age)

